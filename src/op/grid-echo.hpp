#pragma once

#include "grid-base.hpp"

template <int IP, int TP>
struct GridEcho final : SizedGrid<IP, TP>
{
  using typename SizedGrid<IP, TP>::Input;
  using typename SizedGrid<IP, TP>::Output;

  GridEcho(SizedKernel<IP, TP> const *k, Mapping const &mapping, bool const unsafe)
    : SizedGrid<IP, TP>(k, mapping, unsafe)
  {
    assert(this->mapping_.sortedIndices.size() == this->mapping_.cart.size());
    this->workspace_.resize(Sz5{
      mapping.noncartDims[0],
      mapping.echoes,
      mapping.cartDims[0],
      mapping.cartDims[1],
      mapping.cartDims[2]});
  }

  Output A(Index const inChan) const
  {
    Sz5 const dims = this->inputDimensions();
    Index const nC = inChan < 1 ? dims[0] : inChan;

    Sz3 odims = this->outputDimensions();
    odims[0] = nC;
    Output noncart(odims);
    noncart.setZero();

    Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nC);

    auto grid_task = [&](Index const lo, Index const hi) {
      Eigen::IndexList<FixZero, int, int, int> stC;
      for (auto ii = lo; ii < hi; ii++) {
        Log::Progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const e = std::min(this->mapping_.echo[si], int8_t(dims[1] - 1));
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        stC.set(1, c.x - ((IP - 1) / 2));
        stC.set(2, c.y - ((IP - 1) / 2));
        stC.set(3, c.z - ((TP - 1) / 2));
        noncart.template chip<2>(n.spoke).template chip<1>(n.read) =
          this->workspace_.template chip<1>(e).slice(stC, szC).contract(
            k.template cast<Cx>(),
            Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
      }
    };
    auto const &start = Log::Now();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    Log::Debug("Cart -> Non-cart: {}", Log::ToNow(start));
    return noncart;
  }

  void Adj(Output const &noncart, Index const inChan) const
  {
    auto const dims = this->inputDimensions();
    Index const nC = inChan < 1 ? dims[0] : inChan;
    assert(nC == noncart.dimension(0));

    Eigen::IndexList<int, FixOne, FixOne, FixOne> rshNC;
    constexpr Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> brdNC;
    rshNC.set(0, nC);

    constexpr Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> rshK;
    Eigen::IndexList<int, FixOne, FixOne, FixOne> brdK;
    brdK.set(0, nC);

    Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nC);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> threadSpaces(nThreads);
    std::vector<Index> minZ(nThreads, 0L), szZ(nThreads, 0L);
    auto grid_task = [&](Index const lo, Index const hi, Index const ti) {
      // Allocate working space for this thread
      Eigen::IndexList<FixZero, int, int, int> stC;
      minZ[ti] = this->mapping_.cart[this->mapping_.sortedIndices[lo]].z -
                 ((this->kernel_->throughPlane() - 1) / 2);

      if (this->safe_) {
        Index const maxZ = this->mapping_.cart[this->mapping_.sortedIndices[hi - 1]].z +
                           (this->kernel_->throughPlane() / 2);
        szZ[ti] = maxZ - minZ[ti] + 1;
        threadSpaces[ti].resize(nC, dims[1], dims[2], dims[3], szZ[ti]);
        threadSpaces[ti].setZero();
      }

      for (auto ii = lo; ii < hi; ii++) {
        Log::Progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const e = std::min(this->mapping_.echo[si], int8_t(dims[1] - 1));
        auto const sdc = this->weightEchoes_ ? pow(this->mapping_.sdc[si], this->sdcPow_) *
                                                 this->mapping_.echoWeights[e]
                                             : pow(this->mapping_.sdc[si], this->sdcPow_);
        auto const nc = noncart.template chip<2>(n.spoke).template chip<1>(n.read);
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        auto const nck = (nc * nc.constant(sdc)).reshape(rshNC).broadcast(brdNC) *
                         k.template cast<Cx>().reshape(rshK).broadcast(brdK);
        stC.set(1, c.x - ((IP - 1) / 2));
        stC.set(2, c.y - ((IP - 1) / 2));
        if (this->safe_) {
          stC.set(3, c.z - ((TP - 1) / 2) - minZ[ti]);
          threadSpaces[ti].chip<1>(e).slice(stC, szC) += nck;
        } else {
          stC.set(3, c.z - ((TP - 1) / 2));
          this->workspace_.template chip<1>(e).slice(stC, szC) += nck;
        }
      }
    };

    auto const start = Log::Now();
    this->workspace_.setZero();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    Log::Debug("Non-cart -> Cart: {}", Log::ToNow(start));
    if (this->safe_) {
      Log::Print(FMT_STRING("Combining thread workspaces..."));
      auto const start2 = Log::Now();
      Sz5 st{0, 0, 0, 0, 0};
      Sz5 sz{nC, dims[1], dims[2], dims[3], 0};
      for (Index ti = 0; ti < nThreads; ti++) {
        if (szZ[ti]) {
          st[4] = minZ[ti];
          sz[4] = szZ[ti];
          this->workspace_.slice(st, sz).device(dev) += threadSpaces[ti];
        }
      }
      Log::Debug(FMT_STRING("Combining took: {}"), Log::ToNow(start2));
    }
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;
};
