#pragma once

#include "grid-base.hpp"

template <int IP, int TP>
struct GridBasis final : SizedGrid<IP, TP>
{
  using typename SizedGrid<IP, TP>::Input;
  using typename SizedGrid<IP, TP>::Output;

  GridBasis(
    SizedKernel<IP, TP> const *k,
    Mapping const &mapping,
    R2 const &basis,
    bool const unsafe,
    Log &log)
    : SizedGrid<IP, TP>(k, mapping, unsafe, log)
    , basis_{basis}
    , basisScale_{std::sqrt((float)basis_.dimension(0))}
  {
  }

  Index dimension(Index const D) const
  {
    assert(D < 3);
    return this->mapping_.cartDims[D];
  }

  Sz5 inputDimensions(Index const nc) const
  {
    return Sz5{
      nc,
      basis_.dimension(1),
      this->mapping_.cartDims[0],
      this->mapping_.cartDims[1],
      this->mapping_.cartDims[2]};
  }

  R2 const &basis() const
  {
    return basis_;
  }

  void A(Input const &cart, Output &noncart) const
  {
    assert(cart.dimension(0) == noncart.dimension(0));
    assert(cart.dimension(1) == basis_.dimension(1));
    assert(cart.dimension(2) == this->mapping_.cartDims[0]);
    assert(cart.dimension(3) == this->mapping_.cartDims[1]);
    assert(cart.dimension(4) == this->mapping_.cartDims[2]);

    Index const nchan = cart.dimension(0);
    Index const nB = basis_.dimension(1);
    Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);
    szC.set(1, nB);

    auto grid_task = [&](Index const lo, Index const hi) {
      Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
      for (auto ii = lo; ii < hi; ii++) {
        this->log_.progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const b =
          (basis_.chip<0>(n.spoke % basis_.dimension(0)) * basisScale_).template cast<Cx>();
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        stC.set(2, c.x - (IP / 2));
        stC.set(3, c.y - (IP / 2));
        stC.set(4, c.z - (TP / 2));
        noncart.template chip<2>(n.spoke).template chip<1>(n.read) =
          cart.slice(stC, szC)
            .contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>())
            .contract(
              k.template cast<Cx>(),
              Eigen::IndexPairList<
                Eigen::type2indexpair<1, 0>,
                Eigen::type2indexpair<2, 1>,
                Eigen::type2indexpair<3, 2>>());
      }
    };
    auto const &start = this->log_.now();
    noncart.setZero();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    this->log_.debug("Cart -> Non-cart: {}", this->log_.toNow(start));
  }

  void Adj(Output const &noncart, Input &cart) const
  {
    assert(cart.dimension(0) == noncart.dimension(0));
    assert(cart.dimension(1) == basis_.dimension(1));
    assert(cart.dimension(2) == this->mapping_.cartDims[0]);
    assert(cart.dimension(3) == this->mapping_.cartDims[1]);
    assert(cart.dimension(4) == this->mapping_.cartDims[2]);
    assert(this->mapping_.sortedIndices.size() == this->mapping_.cart.size());

    Index const nchan = cart.dimension(0);
    Index const nB = cart.dimension(1);

    Eigen::IndexList<int, FixOne> rshNC;
    Eigen::IndexList<FixOne, int> brdNC;
    rshNC.set(0, nchan);
    brdNC.set(1, nB);

    Eigen::IndexList<FixOne, int> rshB;
    Eigen::IndexList<int, FixOne> brdB;
    rshB.set(1, nB);
    brdB.set(0, nchan);

    constexpr Eigen::IndexList<FixOne, FixOne, FixIn, FixIn, FixThrough> rshK;
    Eigen::IndexList<int, int, FixOne, FixOne, FixOne> brdK;
    brdK.set(0, nchan);
    brdK.set(1, nB);

    Eigen::IndexList<int, int, FixOne, FixOne, FixOne> rshNCB;
    constexpr Eigen::IndexList<FixOne, FixOne, FixIn, FixIn, FixThrough> brdNCB;
    rshNCB.set(0, nchan);
    rshNCB.set(1, nB);

    Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);
    szC.set(1, nB);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> workspace(nThreads);
    std::vector<Index> minZ(nThreads, 0L), szZ(nThreads, 0L);
    auto grid_task = [&](Index const lo, Index const hi, Index const ti) {
      // Allocate working space for this thread
      Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
      Cx2 ncb(nchan, nB);

      minZ[ti] = this->mapping_.cart[this->mapping_.sortedIndices[lo]].z - ((TP - 1) / 2);

      if (this->safe_) {
        Index const maxZ = this->mapping_.cart[this->mapping_.sortedIndices[hi - 1]].z + (TP / 2);
        szZ[ti] = maxZ - minZ[ti] + 1;
        workspace[ti].resize(
          cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]);
        workspace[ti].setZero();
      }

      for (auto ii = lo; ii < hi; ii++) {
        this->log_.progress(ii, lo, hi);
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const sdc = pow(this->mapping_.sdc[si], this->sdcPow_);
        auto const nc = noncart.template chip<2>(n.spoke).template chip<1>(n.read);
        auto const b = (basis_.chip<0>(n.spoke % basis_.dimension(0)) * basisScale_);
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        ncb = (nc * nc.constant(sdc)).reshape(rshNC).broadcast(brdNC) *
              b.template cast<Cx>().reshape(rshB).broadcast(brdB);
        auto const nbk = ncb.reshape(rshNCB).broadcast(brdNCB) *
                         k.template cast<Cx>().reshape(rshK).broadcast(brdK);

        stC.set(2, c.x - (IP / 2));
        stC.set(3, c.y - (IP / 2));
        if (this->safe_) {
          stC.set(4, c.z - (TP / 2) - minZ[ti]);
          workspace[ti].slice(stC, szC) += nbk;
        } else {
          stC.set(4, c.z - (TP / 2));
          cart.slice(stC, szC) += nbk;
        }
      }
    };

    auto const start = this->log_.now();
    cart.setZero();
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    this->log_.debug("Basis Non-cart -> Cart: {}", this->log_.toNow(start));
    if (this->safe_) {
      this->log_.info("Combining thread workspaces...");
      auto const start2 = this->log_.now();
      for (Index ti = 0; ti < nThreads; ti++) {
        if (szZ[ti]) {
          cart
            .slice(
              Sz5{0, 0, 0, 0, minZ[ti]},
              Sz5{
                cart.dimension(0),
                cart.dimension(1),
                cart.dimension(2),
                cart.dimension(3),
                szZ[ti]})
            .device(dev) += workspace[ti];
        }
      }
      this->log_.debug("Combining took: {}", this->log_.toNow(start2));
    }
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;

  R2 basis_;
  float basisScale_;
};
