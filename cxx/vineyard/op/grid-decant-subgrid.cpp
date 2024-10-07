#include "grid-decant-subgrid.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"

namespace rl {

template <> void GridToDecant<1>(Sz1 const sg, Cx3 const &sk, Cx2CMap const &x, Cx3 &sx)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    for (Index kx = 0; kx < sk.dimension(2); kx++) {
      Index const kkx = sk.dimension(2) - 1 - kx; // Reverse kernel
      Index const ox = kx - sk.dimension(2) / 2;
      Index const iix = Wrap(ix + ox + sg[0], x.dimension(1));
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        if (sk.dimension(0) == 1) {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            sx(ib, ic, ix) = x(ib, iix) * sk(0, ic, kkx);
          }
        } else {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            sx(ib, ic, ix) = x(ib, iix) * sk(ib, ic, kkx);
          }
        }
      }
    }
  }
}

template <> void GridToDecant<2>(Sz2 const sg, Cx4 const &sk, Cx3CMap const &x, Cx4 &sx)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    for (Index ky = 0; ky < sk.dimension(3); ky++) {
      Index const kky = sk.dimension(3) - 1 - ky; // Reverse kernel
      Index const oy = ky - sk.dimension(3) / 2;
      Index const iiy = Wrap(iy + oy + sg[1], x.dimension(2));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        for (Index kx = 0; kx < sk.dimension(2); kx++) {
          Index const kkx = sk.dimension(2) - 1 - kx; // Reverse kernel
          Index const ox = kx - sk.dimension(2) / 2;
          Index const iix = Wrap(ix + ox + sg[0], x.dimension(1));
          for (Index ic = 0; ic < sx.dimension(1); ic++) {
            if (sk.dimension(0) == 1) {
              for (Index ib = 0; ib < sx.dimension(0); ib++) {
                sx(ib, ic, ix, iy) = x(ib, iix, iiy) * sk(0, ic, kkx, kky);
              }
            } else {
              for (Index ib = 0; ib < sx.dimension(0); ib++) {
                sx(ib, ic, ix, iy) = x(ib, iix, iiy) * sk(ib, ic, kkx, kky);
              }
            }
          }
        }
      }
    }
  }
}

template <> void GridToDecant<3>(Sz3 const sg, Cx5 const &sk, Cx4CMap const &x, Cx5 &sx)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    for (Index kz = 0; kz < sk.dimension(4); kz++) {
      Index const kkz = sk.dimension(4) - 1 - kz;
      Index const oz = kz - sk.dimension(4) / 2;
      Index const iiz = Wrap(iz + oz + sg[2], x.dimension(3));
      for (Index iy = 0; iy < sx.dimension(3); iy++) {
        for (Index ky = 0; ky < sk.dimension(3); ky++) {
          Index const kky = sk.dimension(3) - 1 - ky; // Reverse kernel
          Index const oy = ky - sk.dimension(3) / 2;
          Index const iiy = Wrap(iy + oy + sg[1], x.dimension(2));
          for (Index ix = 0; ix < sx.dimension(2); ix++) {
            for (Index kx = 0; kx < sk.dimension(2); kx++) {
              Index const kkx = sk.dimension(2) - 1 - kx; // Reverse kernel
              Index const ox = kx - sk.dimension(2) / 2;
              Index const iix = Wrap(ix + ox + sg[0], x.dimension(1));
              for (Index ic = 0; ic < sx.dimension(1); ic++) {
                if (sk.dimension(0) == 1) {
                  for (Index ib = 0; ib < sx.dimension(0); ib++) {
                    sx(ib, ic, ix, iy, iz) = x(ib, iix, iiy, iiz) * sk(0, ic, kkx, kky, kkz);
                  }
                } else {
                  for (Index ib = 0; ib < sx.dimension(0); ib++) {
                    sx(ib, ic, ix, iy, iz) = x(ib, iix, iiy, iiz) * sk(ib, ic, kkx, kky, kkz);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <> void DecantToGrid<1>(std::vector<std::mutex> &m, Sz1 const sg, Cx3 const &sk, Cx3CMap const &sx, Cx2Map &x)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  assert(m.size() == x.dimension(1));
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    for (Index kx = 0; kx < sk.dimension(2); kx++) {
      Index const      kkx = sk.dimension(2) - 1 - kx; // Still reverse kernel as adjoint
      Index const      ox = kx - sk.dimension(2) / 2;
      Index const      iix = Wrap(ix + ox + sg[0], x.dimension(1));
      std::scoped_lock lock(m[iix]);
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        if (sk.dimension(0) == 1) {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            x(ib, iix) += sx(ib, ic, ix) * std::conj(sk(0, ic, kkx));
          }
        } else {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            x(ib, iix) += sx(ib, ic, ix) * std::conj(sk(ib, ic, kkx));
          }
        }
      }
    }
  }
}

template <> void DecantToGrid<2>(std::vector<std::mutex> &m, Sz2 const sg, Cx4 const &sk, Cx4CMap const &sx, Cx3Map &x)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  assert(m.size() == x.dimension(2));
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    for (Index ky = 0; ky < sk.dimension(3); ky++) {
      Index const      kky = sk.dimension(3) - 1 - ky;
      Index const      oy = ky - sk.dimension(3) / 2;
      Index const      iiy = Wrap(iy + oy + sg[1], x.dimension(2));
      std::scoped_lock lock(m[iiy]);
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        for (Index kx = 0; kx < sk.dimension(2); kx++) {
          Index const kkx = sk.dimension(2) - 1 - kx; // Still reverse kernel as adjoint
          Index const ox = kx - sk.dimension(2) / 2;
          Index const iix = Wrap(ix + ox + sg[0], x.dimension(1));
          for (Index ic = 0; ic < sx.dimension(1); ic++) {
            if (sk.dimension(0) == 1) {
              for (Index ib = 0; ib < sx.dimension(0); ib++) {
                x(ib, iix, iiy) += sx(ib, ic, ix, iy) * std::conj(sk(0, ic, kkx, kky));
              }
            } else {
              for (Index ib = 0; ib < sx.dimension(0); ib++) {
                x(ib, iix, iiy) += sx(ib, ic, ix, iy) * std::conj(sk(ib, ic, kkx, kky));
              }
            }
          }
        }
      }
    }
  }
}

template <> void DecantToGrid<3>(std::vector<std::mutex> &m, Sz3 const sg, Cx5 const &sk, Cx5CMap const &sx, Cx4Map &x)
{
  assert(sk.dimension(0) == sx.dimension(0));
  assert(sk.dimension(1) == sx.dimension(1));
  assert(m.size() == x.dimension(3));
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    for (Index kz = 0; kz < sk.dimension(4); kz++) {
      Index const kkz = sk.dimension(4) - 1 - kz;
      Index const oz = kz - sk.dimension(4) / 2;
      Index const iiz = Wrap(iz + oz + sg[2], x.dimension(3));
      std::scoped_lock lock(m[iiz]);
      for (Index iy = 0; iy < sx.dimension(3); iy++) {
        for (Index ky = 0; ky < sk.dimension(3); ky++) {
          Index const kky = sk.dimension(3) - 1 - ky;
          Index const oy = ky - sk.dimension(3) / 2;
          Index const iiy = Wrap(iy + oy + sg[1], x.dimension(2));
          for (Index ix = 0; ix < sx.dimension(2); ix++) {
            for (Index kx = 0; kx < sk.dimension(2); kx++) {
              Index const kkx = sk.dimension(2) - 1 - kx; // Still reverse kernel as adjoint
              Index const ox = kx - sk.dimension(2) / 2;
              Index const iix = Wrap(ix + ox + sg[0], x.dimension(1));
              for (Index ic = 0; ic < sx.dimension(1); ic++) {
                if (sk.dimension(0) == 1) {
                  for (Index ib = 0; ib < sx.dimension(0); ib++) {
                    x(ib, iix, iiy, iiz) += sx(ib, ic, ix, iy, iz) * std::conj(sk(0, ic, kkx, kky, kkz));
                  }
                } else {
                  for (Index ib = 0; ib < sx.dimension(0); ib++) {
                    x(ib, iix, iiy, iiz) += sx(ib, ic, ix, iy, iz) * std::conj(sk(ib, ic, kkx, kky, kkz));
                  }
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
