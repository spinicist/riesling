#pragma once

#include "../types.hpp"
#include <atomic>

namespace rl {

template <int ND, int SGSZ> struct DecantSubgrid
{
};

template <int SGSZ> struct DecantSubgrid<1, SGSZ>
{
  inline static void FastForward(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap sk, Cx2CMap x, Cx3Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx;
          Index const ox = kx - sk.dimension(0) / 2;
          for (Index ix = 0; ix < sx.dimension(0); ix++) {
            Index const iix = ix + ox + sg[0];
            sx(ix, ib, ic) += x(iix, ib) * sk(kkx, ib, ic);
          }
        }
      }
    }
  }

  inline static void SlowForward(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap sk, Cx2CMap x, Cx3Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx;
          Index const ox = kx - sk.dimension(0) / 2;
          for (Index ix = 0; ix < sx.dimension(0); ix++) {
            Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
            sx(ix, ib, ic) += x(iix, ib) * sk(kkx, ib, ic);
          }
        }
      }
    }
  }

  inline static void
  FastAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap sk, Cx3CMap sx, Cx2Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
          Index const ox = kx - sk.dimension(0) / 2;
          for (Index ix = 0; ix < SGSZ; ix++) {
            Index const      iix = ix + ox + corner[0];
            std::scoped_lock lock(m[iix]);
            x(iix, ib) += sx(ix, ib, ic) * std::conj(sk(kkx, ib, ic));
            ;
          }
        }
      }
    }
  }

  inline static void
  SlowAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap sk, Cx3CMap sx, Cx2Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index kx = 0; kx < sk.dimension(0); kx++) {
          Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
          Index const ox = kx - sk.dimension(0) / 2;
          for (Index ix = 0; ix < SGSZ; ix++) {
            Index const      iix = Wrap(ix + ox + corner[0], x.dimension(0));
            std::scoped_lock lock(m[iix]);
            x(iix, ib) += sx(ix, ib, ic) * std::conj(sk(kkx, ib, ic));
          }
        }
      }
    }
  }
};

template <int SGSZ> struct DecantSubgrid<2, SGSZ>
{
  inline static void FastForward(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap sk, Cx3CMap x, Cx4Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ky = 0; ky < sk.dimension(1); ky++) {
          Index const kky = sk.dimension(1) - 1 - ky;
          Index const oy = ky - sk.dimension(1) / 2;
          for (Index kx = 0; kx < sk.dimension(0); kx++) {
            Index const kkx = sk.dimension(0) - 1 - kx;
            Index const ox = kx - sk.dimension(0) / 2;
            for (Index iy = 0; iy < sx.dimension(1); iy++) {
              Index const iiy = iy + oy + sg[1];
              for (Index ix = 0; ix < sx.dimension(0); ix++) {
                Index const iix = ix + ox + sg[0];
                sx(ix, iy, ib, ic) += x(iix, iiy, ib) * sk(kkx, kky, ib, ic);
              }
            }
          }
        }
      }
    }
  }

  inline static void SlowForward(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap sk, Cx3CMap x, Cx4Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ky = 0; ky < sk.dimension(1); ky++) {
          Index const kky = sk.dimension(1) - 1 - ky;
          Index const oy = ky - sk.dimension(1) / 2;
          for (Index kx = 0; kx < sk.dimension(0); kx++) {
            Index const kkx = sk.dimension(0) - 1 - kx;
            Index const ox = kx - sk.dimension(0) / 2;
            for (Index iy = 0; iy < sx.dimension(1); iy++) {
              Index const iiy = Wrap(iy + oy + sg[1], x.dimension(1));
              for (Index ix = 0; ix < sx.dimension(0); ix++) {
                Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
                sx(ix, iy, ib, ic) += x(iix, iiy, ib) * sk(kkx, kky, ib, ic);
              }
            }
          }
        }
      }
    }
  }

  inline static void
  FastAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap sk, Cx4CMap sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ky = 0; ky < sk.dimension(1); ky++) {
          Index const kky = sk.dimension(1) - 1 - ky;
          Index const oy = ky - sk.dimension(1) / 2;
          for (Index kx = 0; kx < sk.dimension(0); kx++) {
            Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
            Index const ox = kx - sk.dimension(0) / 2;
            for (Index iy = 0; iy < SGSZ; iy++) {
              Index const      iiy = iy + oy + corner[1];
              std::scoped_lock lock(m[iiy]);
              for (Index ix = 0; ix < SGSZ; ix++) {
                Index const iix = ix + ox + corner[0];
                x(iix, iiy, ib) += sx(ix, iy, ib, ic) * std::conj(sk(kkx, kky, ib, ic));
              }
            }
          }
        }
      }
    }
  }

  inline static void
  SlowAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap sk, Cx4CMap sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ky = 0; ky < sk.dimension(1); ky++) {
          Index const kky = sk.dimension(1) - 1 - ky;
          Index const oy = ky - sk.dimension(1) / 2;
          for (Index kx = 0; kx < sk.dimension(0); kx++) {
            Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
            Index const ox = kx - sk.dimension(0) / 2;
            for (Index iy = 0; iy < SGSZ; iy++) {
              Index const      iiy = Wrap(iy + oy + corner[1], x.dimension(1));
              std::scoped_lock lock(m[iiy]);
              for (Index ix = 0; ix < SGSZ; ix++) {
                Index const iix = Wrap(ix + ox + corner[0], x.dimension(0));
                x(iix, iiy, ib) += sx(ix, iy, ib, ic) * std::conj(sk(kkx, kky, ib, ic));
              }
            }
          }
        }
      }
    }
  }
};

template <int SGSZ> struct DecantSubgrid<3, SGSZ>
{
  inline static void FastForward(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap sk, Cx4CMap x, Cx5Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index kz = 0; kz < sk.dimension(2); kz++) {
          Index const kkz = sk.dimension(2) - 1 - kz;
          Index const oz = kz - sk.dimension(2) / 2;
          for (Index ky = 0; ky < sk.dimension(1); ky++) {
            Index const kky = sk.dimension(1) - 1 - ky;
            Index const oy = ky - sk.dimension(1) / 2;
            for (Index kx = 0; kx < sk.dimension(0); kx++) {
              Index const kkx = sk.dimension(0) - 1 - kx;
              Index const ox = kx - sk.dimension(0) / 2;
              for (Index iz = 0; iz < sx.dimension(2); iz++) {
                Index const iiz = iz + oz + sg[2];
                for (Index iy = 0; iy < sx.dimension(1); iy++) {
                  Index const iiy = iy + oy + sg[1];
                  for (Index ix = 0; ix < sx.dimension(0); ix++) {
                    Index const iix = ix + ox + sg[0];
                    sx(ix, iy, iz, ib, ic) += x(iix, iiy, iiz, ib) * sk(kkx, kky, kkz, ib, ic);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  inline static void SlowForward(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap sk, Cx4CMap x, Cx5Map sx)
  {
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index kz = 0; kz < sk.dimension(2); kz++) {
          Index const kkz = sk.dimension(2) - 1 - kz;
          Index const oz = kz - sk.dimension(2) / 2;
          for (Index ky = 0; ky < sk.dimension(1); ky++) {
            Index const kky = sk.dimension(1) - 1 - ky;
            Index const oy = ky - sk.dimension(1) / 2;
            for (Index kx = 0; kx < sk.dimension(0); kx++) {
              Index const kkx = sk.dimension(0) - 1 - kx;
              Index const ox = kx - sk.dimension(0) / 2;
              for (Index iz = 0; iz < sx.dimension(2); iz++) {
                Index const iiz = Wrap(iz + oz + sg[2], x.dimension(2));
                for (Index iy = 0; iy < sx.dimension(1); iy++) {
                  Index const iiy = Wrap(iy + oy + sg[1], x.dimension(1));
                  for (Index ix = 0; ix < sx.dimension(0); ix++) {
                    Index const iix = Wrap(ix + ox + sg[0], x.dimension(0));
                    sx(ix, iy, iz, ib, ic) += x(iix, iiy, iiz, ib) * sk(kkx, kky, kkz, ib, ic);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  inline static void
  FastAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap sk, Cx5CMap sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index kz = 0; kz < sk.dimension(2); kz++) {
          Index const kkz = sk.dimension(2) - 1 - kz;
          Index const oz = kz - sk.dimension(2) / 2;
          for (Index ky = 0; ky < sk.dimension(1); ky++) {
            Index const kky = sk.dimension(1) - 1 - ky;
            Index const oy = ky - sk.dimension(1) / 2;
            for (Index kx = 0; kx < sk.dimension(0); kx++) {
              Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
              Index const ox = kx - sk.dimension(0) / 2;
              for (Index iz = 0; iz < SGSZ; iz++) {
                Index const iiz = iz + oz + corner[2];
                for (Index iy = 0; iy < SGSZ; iy++) {
                  Index const iiy = iy + oy + corner[1];
                  for (Index ix = 0; ix < SGSZ; ix++) {
                    Index const         iix = ix + ox + corner[0];
                    std::atomic_ref<Cx> ref(x(iix, iiy, iiz, ib));
                    ref = x(iix, iiy, iiz, ib) + sx(ix, iy, iz, ib, ic) * std::conj(sk(kkx, kky, kkz, ib, ic));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  inline static void
  SlowAdjoint(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap sk, Cx5CMap sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index kz = 0; kz < sk.dimension(2); kz++) {
          Index const kkz = sk.dimension(2) - 1 - kz;
          Index const oz = kz - sk.dimension(2) / 2;
          for (Index ky = 0; ky < sk.dimension(1); ky++) {
            Index const kky = sk.dimension(1) - 1 - ky;
            Index const oy = ky - sk.dimension(1) / 2;
            for (Index kx = 0; kx < sk.dimension(0); kx++) {
              Index const kkx = sk.dimension(0) - 1 - kx; // Still reverse kernel as adjoint
              Index const ox = kx - sk.dimension(0) / 2;
              for (Index iz = 0; iz < SGSZ; iz++) {
                Index const iiz = Wrap(iz + oz + corner[2], x.dimension(2));
                for (Index iy = 0; iy < SGSZ; iy++) {
                  Index const iiy = Wrap(iy + oy + corner[1], x.dimension(1));
                  for (Index ix = 0; ix < SGSZ; ix++) {
                    Index const         iix = Wrap(ix + ox + corner[0], x.dimension(0));
                    std::atomic_ref<Cx> ref(x(iix, iiy, iiz, ib));
                    ref = x(iix, iiy, iiz, ib) + sx(ix, iy, iz, ib, ic) * std::conj(sk(kkx, kky, kkz, ib, ic));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

} // namespace rl
