#include "grid-decant-subgrid.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"

namespace rl {

template <> void GridToDecant<1>(Eigen::Array<int16_t, 1, 1> const sg, Cx3 const &sk, Cx2CMap const &x, Cx3 &sx)
{
  assert(sk.dimension(1) == sx.dimension(1));
  assert(sk.dimension(2) == sx.dimension(2));
  for (Index ix = 0; ix < sx.dimension(0); ix++) {
    for (Index kx = 0; kx < sk.dimension(0); kx++) {
      Index const kkx = sk.dimension(0) - 1 - kx; // Reverse kernel
      Index const ox = kx - sk.dimension(0) / 2;
      Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        for (Index ib = 0; ib < sx.dimension(2); ib++) {
          sx(ix, ic, ib) = x(iix, ib) * sk(kkx, ic, ib);
        }
      }
    }
  }
}

template <> void GridToDecant<2>(Eigen::Array<int16_t, 2, 1> const sg, Cx4 const &sk, Cx3CMap const &x, Cx4 &sx)
{
  assert(sk.dimension(2) == sx.dimension(2));
  assert(sk.dimension(3) == sx.dimension(3));
  for (Index iy = 0; iy < sx.dimension(1); iy++) {
    for (Index ky = 0; ky < sk.dimension(1); ky++) {
      Index const kky = sk.dimension(1) - 1 - ky; // Reverse kernel
      Index const oy = ky - sk.dimension(1) / 2;
      Index const iiy = Wrap(iy + oy + sg[1], x.dimension(1));
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx; // Reverse kernel
          Index const ox = kx - sk.dimension(0) / 2;
          Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            for (Index ib = 0; ib < sx.dimension(3); ib++) {
              sx(ix, iy, ic, ib) = x(iix, iiy, ib) * sk(kkx, kky, ic, ib);
            }
          }
        }
      }
    }
  }
}

template <> void GridToDecant<3>(Eigen::Array<int16_t, 3, 1> const sg, Cx5 const &sk, Cx4CMap const &x, Cx5 &sx)
{
  assert(sk.dimension(3) == sx.dimension(3));
  assert(sk.dimension(4) == sx.dimension(4));
  for (Index iz = 0; iz < sx.dimension(2); iz++) {
    for (Index kz = 0; kz < sk.dimension(2); kz++) {
      Index const kkz = sk.dimension(2) - 1 - kz;
      Index const oz = kz - sk.dimension(2) / 2;
      Index const iiz = Wrap(iz + oz + sg[2], x.dimension(2));
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        for (Index ky = 0; ky < sk.dimension(1); ky++) {
          Index const kky = sk.dimension(1) - 1 - ky; // Reverse kernel
          Index const oy = ky - sk.dimension(1) / 2;
          Index const iiy = Wrap(iy + oy + sg[1], x.dimension(1));
          for (Index ix = 0; ix < sx.dimension(0); ix++) {
            for (Index kx = 0; kx < sk.dimension(0); kx++) {
              Index const kkx = sk.dimension(0) - 1 - kx; // Reverse kernel
              Index const ox = kx - sk.dimension(0) / 2;
              Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
              for (Index ic = 0; ic < sx.dimension(3); ic++) {
                for (Index ib = 0; ib < sx.dimension(4); ib++) {
                  sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ib) * sk(kkx, kky, kkz, ic, ib);
                }
              }
            }
          }
        }
      }
    }
  }
}

template <>
void DecantToGrid<1>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const sg, Cx3 const &sk, Cx3CMap const &sx, Cx2Map &x)
{
  assert(sk.dimension(1) == sx.dimension(1));
  assert(sk.dimension(2) == sx.dimension(2));
  assert(m.size() == x.dimension(0));
  for (Index ix = 0; ix < sx.dimension(0); ix++) {
    for (Index kx = 0; kx < sk.dimension(0); kx++) {
      Index const      kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
      Index const      ox = kx - sk.dimension(0) / 2;
      Index const      iix = Wrap(ix + ox + sg[0], x.dimension(0));
      std::scoped_lock lock(m[iix]);
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        for (Index ib = 0; ib < sx.dimension(2); ib++) {
          x(iix, ib) += sx(ix, ic, ib) * std::conj(sk(kkx, ic, ib));
        }
      }
    }
  }
}

template <>
void DecantToGrid<2>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const sg, Cx4 const &sk, Cx4CMap const &sx, Cx3Map &x)
{
  assert(sk.dimension(1) == sx.dimension(1));
  assert(sk.dimension(2) == sx.dimension(2));
  assert(m.size() == x.dimension(1));
  for (Index iy = 0; iy < sx.dimension(1); iy++) {
    for (Index ky = 0; ky < sk.dimension(1); ky++) {
      Index const      kky = sk.dimension(1) - 1 - ky;
      Index const      oy = ky - sk.dimension(1) / 2;
      Index const      iiy = Wrap(iy + oy + sg[1], x.dimension(1));
      std::scoped_lock lock(m[iiy]);
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
          Index const ox = kx - sk.dimension(0) / 2;
          Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            for (Index ib = 0; ib < sx.dimension(0); ib++) {
              x(iix, iiy, ib) += sx(ix, iy, ic, ib) * std::conj(sk(kkx, kky, ic, ib));
            }
          }
        }
      }
    }
  }
}

template <>
void DecantToGrid<3>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const sg, Cx5 const &sk, Cx5CMap const &sx, Cx4Map &x)
{
  assert(sk.dimension(2) == sx.dimension(2));
  assert(sk.dimension(3) == sx.dimension(3));
  assert(m.size() == x.dimension(2));
  Index const nB = sx.dimension(4);
  Index const nC = sx.dimension(3);
  for (Index kz = 0; kz < sk.dimension(2); kz++) {
    Index const kkz = sk.dimension(2) - 1 - kz;
    Index const oz = kz - sk.dimension(2) / 2;
    for (Index ky = 0; ky < sk.dimension(1); ky++) {
      Index const kky = sk.dimension(1) - 1 - ky;
      Index const oy = ky - sk.dimension(1) / 2;
      for (Index kx = 0; kx < sk.dimension(0); kx++) {
        Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
        Index const ox = kx - sk.dimension(0) / 2;
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const      iiz = Wrap(iz + oz + sg[2], x.dimension(2));
          std::scoped_lock lock(m[iiz]);
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = Wrap(iy + oy + sg[1], x.dimension(1));
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
              for (Index ic = 0; ic < nC; ic++) {
                for (Index ib = 0; ib < nB; ib++) {
                  x(iix, iiy, iiz, ib) += sx(ix, iy, iz, ic, ib) * std::conj(sk(kkx, kky, kkz, ic, ib));
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace rl
