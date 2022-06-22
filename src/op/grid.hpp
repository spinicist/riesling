#pragma once

#include "grid-base.hpp"

template <int IP, int TP>
struct Grid final : SizedGrid<IP, TP>
{
  using typename SizedGrid<IP, TP>::Input;
  using typename SizedGrid<IP, TP>::Output;

  Grid(SizedKernel<IP, TP> const *k, Mapping const &mapping, bool const unsafe, std::shared_ptr<Cx5> ws)
    : SizedGrid<IP, TP>(k, mapping, mapping.frames, unsafe, ws)
  {
    Log::Debug(FMT_STRING("Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Output A(Input const &cart) const
  {
    Sz5 const cdims = cart.dimensions();
    Index const nC = cart.dimension(0);
    if (LastN<4>(cdims) != LastN<4>(this->inputDimensions())) {
      Log::Fail(FMT_STRING("Cartesian k-space dims {} did not match {}"), cdims, this->inputDimensions());
    }
    Sz3 ncdims = this->outputDimensions();
    ncdims[0] = nC;
    Output noncart(ncdims);
    noncart.setZero();

    auto const scale = this->mapping_.scale;
    auto grid_task = [&](Index const ii) {
      auto const si = this->mapping_.sortedIndices[ii];
      auto const c = this->mapping_.cart[si];
      auto const n = this->mapping_.noncart[si];
      auto const ifr = this->mapping_.frame[si];
      auto const k = this->kernel_->k(this->mapping_.offset[si]);
      Index const stX = c.x - ((IP - 1) / 2);
      Index const stY = c.y - ((IP - 1) / 2);
      Index const stZ = c.z - ((TP - 1) / 2);
      Cx1 sum(nC);
      sum.setZero();
      for (Index iz = 0; iz < TP; iz++) {
        for (Index iy = 0; iy < IP; iy++) {
          for (Index ix = 0; ix < IP; ix++) {
            float const kval = k(ix, iy, iz) * scale;
            for (Index ic = 0; ic < nC; ic++) {
              sum(ic) += cart(ic, ifr, stX + ix, stY + iy, stZ + iz) * kval;
            }
          }
        }
      }
      noncart.chip(n.spoke, 2).chip(n.read, 1) = sum;
    };
    auto const &start = Log::Now();
    Threads::For(grid_task, this->mapping_.cart.size(), "Forward Gridding");
    Log::Debug("Cart -> Non-cart: {}", Log::ToNow(start));
    return noncart;
  }

  Input Adj(Output const &noncart) const
  {
    Log::Debug("Grid Adjoint");
    auto const ncdims = noncart.dimensions();
    Index const nC = ncdims[0];
    if (LastN<2>(ncdims) != LastN<2>(this->outputDimensions())) {
      Log::Fail(FMT_STRING("Noncartesian k-space dims {} did not match {}"), ncdims, this->outputDimensions());
    }
    auto cdims = this->inputDimensions();
    if (nC > cdims[0]) {
      Log::Fail(FMT_STRING("Request more channels {} than workspace channels {}"), nC, cdims[0]);
    }
    cdims[0] = nC;
    Input &cart = *(this->ws_);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> threadSpaces(nThreads);
    std::vector<Index> minZ(nThreads, 0L), szZ(nThreads, 0L);
    auto grid_task = [&](Index const lo, Index const hi, Index const ti) {
      if (this->safe_) {
        Index const maxZ = this->mapping_.cart[this->mapping_.sortedIndices[hi - 1]].z + (TP / 2);
        minZ[ti] = this->mapping_.cart[this->mapping_.sortedIndices[lo]].z - ((TP - 1) / 2);
        szZ[ti] = maxZ - minZ[ti] + 1;
        threadSpaces[ti].resize(nC, cdims[1], cdims[2], cdims[3], szZ[ti]);
        threadSpaces[ti].setZero();
      }
      Cx5 &out = this->safe_ ? threadSpaces[ti] : cart;

      for (auto ii = lo; ii < hi; ii++) {
        if (ti == 0) {
          Log::Tick();
        }
        auto const si = this->mapping_.sortedIndices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const ifr = this->mapping_.frame[si];
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        auto const scale = this->mapping_.scale * (this->weightFrames_ ? this->mapping_.frameWeights[ifr] : 1.f);

        Index const stX = c.x - ((IP - 1) / 2);
        Index const stY = c.y - ((IP - 1) / 2);
        Index const stZ = c.z - ((TP - 1) / 2) - minZ[ti];
        for (Index iz = 0; iz < TP; iz++) {
          for (Index iy = 0; iy < IP; iy++) {
            for (Index ix = 0; ix < IP; ix++) {
              float const kval = k(ix, iy, iz) * scale;
              for (Index ic = 0; ic < nC; ic++) {
                out(ic, ifr, stX + ix, stY + iy, stZ + iz) += noncart(ic, n.read, n.spoke) * kval;
              }
            }
          }
        }
      }
    };

    auto const start = Log::Now();
    cart.setZero();
    Log::StartProgress(this->mapping_.cart.size() / dev.numThreads(), "Adjoint Gridding");
    Threads::RangeFor(grid_task, this->mapping_.cart.size());
    Log::StopProgress();
    Log::Debug("Grid Adjoint took: {}", Log::ToNow(start));
    if (this->safe_) {
      Log::Debug(FMT_STRING("Combining thread workspaces..."));
      auto const start2 = Log::Now();
      Sz5 st{0, 0, 0, 0, 0};
      Sz5 sz{nC, cdims[1], cdims[2], cdims[3], 0};
      for (Index ti = 0; ti < nThreads; ti++) {
        if (szZ[ti]) {
          st[4] = minZ[ti];
          sz[4] = szZ[ti];
          cart.slice(st, sz).device(dev) += threadSpaces[ti];
        }
      }
      Log::Debug(FMT_STRING("Combining took: {}"), Log::ToNow(start2));
    }

    if (nC == this->inputDimensions()[0]) {
      return cart;
    } else {
      return cart.slice(Sz5{0, 0, 0, 0, 0}, Sz5{nC, cdims[1], cdims[2], cdims[3], cdims[4]});
    }
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;
};
