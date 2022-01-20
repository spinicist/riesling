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
  }

  Sz5 inputDimensions(Index const nc) const
  {
    return Sz5{
      nc,
      this->mapping_.echoes,
      this->mapping_.cartDims[0],
      this->mapping_.cartDims[1],
      this->mapping_.cartDims[2]};
  }

  Sz5 inputDimensions(Index const nc, Index const ne) const
  {
    return Sz5{
      nc, ne, this->mapping_.cartDims[0], this->mapping_.cartDims[1], this->mapping_.cartDims[2]};
  }

  void A(Input const &cart, Output &noncart) const
  {
    assert(noncart.dimension(0) == cart.dimension(0));
    assert(cart.dimension(2) == this->mapping_.cartDims[0]);
    assert(cart.dimension(3) == this->mapping_.cartDims[1]);
    assert(cart.dimension(4) == this->mapping_.cartDims[2]);

    Index const nchan = cart.dimension(0);
    Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);

    auto grid_task = [&](Index const lo, Index const hi) {
      Eigen::IndexList<FixZero, int, int, int> stC;
      for (auto ii = lo; ii < hi; ii++) {
        Log::Progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const e = std::min(this->mapping_.echo[si], int8_t(cart.dimension(1) - 1));
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        stC.set(1, c.x - (IP / 2));
        stC.set(2, c.y - (IP / 2));
        stC.set(3, c.z - (TP / 2));
        noncart.template chip<2>(n.spoke).template chip<1>(n.read) =
          cart.template chip<1>(e).slice(stC, szC).contract(
            k.template cast<Cx>(),
            Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
      }
    };
    auto const &start = Log::Now();
    noncart.setZero();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    Log::Debug("Cart -> Non-cart: {}", Log::ToNow(start));
  }

  void Adj(Output const &noncart, Input &cart) const
  {
    assert(noncart.dimension(0) == cart.dimension(0));
    assert(cart.dimension(2) == this->mapping_.cartDims[0]);
    assert(cart.dimension(3) == this->mapping_.cartDims[1]);
    assert(cart.dimension(4) == this->mapping_.cartDims[2]);
    assert(this->mapping_.sortedIndices.size() == this->mapping_.cart.size());

    Index const nchan = cart.dimension(0);
    Eigen::IndexList<int, FixOne, FixOne, FixOne> rshNC;
    constexpr Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> brdNC;
    rshNC.set(0, nchan);

    constexpr Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> rshK;
    Eigen::IndexList<int, FixOne, FixOne, FixOne> brdK;
    brdK.set(0, nchan);

    Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> workspace(nThreads);
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
        workspace[ti].resize(
          cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]);
        workspace[ti].setZero();
      }

      for (auto ii = lo; ii < hi; ii++) {
        Log::Progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const e = std::min(this->mapping_.echo[si], int8_t(cart.dimension(1) - 1));
        auto const sdc = this->weightEchoes_ ? pow(this->mapping_.sdc[si], this->sdcPow_) *
                                                 this->mapping_.echoWeights[e]
                                             : pow(this->mapping_.sdc[si], this->sdcPow_);
        auto const nc = noncart.template chip<2>(n.spoke).template chip<1>(n.read);
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        auto const nck = (nc * nc.constant(sdc)).reshape(rshNC).broadcast(brdNC) *
                         k.template cast<Cx>().reshape(rshK).broadcast(brdK);
        stC.set(1, c.x - (IP / 2));
        stC.set(2, c.y - (IP / 2));
        if (this->safe_) {
          stC.set(3, c.z - (TP / 2) - minZ[ti]);
          workspace[ti].chip<1>(e).slice(stC, szC) += nck;
        } else {
          stC.set(3, c.z - (TP / 2));
          cart.template chip<1>(e).slice(stC, szC) += nck;
        }
      }
    };

    auto const start = Log::Now();
    cart.setZero();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    Log::Debug("Non-cart -> Cart: {}", Log::ToNow(start));
    if (this->safe_) {
      Log::Print("Combining thread workspaces...");
      auto const start2 = Log::Now();
      Sz5 st{0, 0, 0, 0, 0};
      Sz5 sz = cart.dimensions();
      for (Index ti = 0; ti < nThreads; ti++) {
        if (szZ[ti]) {
          st[4] = minZ[ti];
          sz[4] = szZ[ti];
          cart.slice(st, sz).device(dev) += workspace[ti];
        }
      }
      Log::Debug("Combining took: {}", Log::ToNow(start2));
    }
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;
};
