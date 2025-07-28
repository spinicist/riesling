#pragma once

#include "../types.hpp"
#include <mutex>

namespace rl {

template <int ND, int SGSZ> struct GridToSubgrid
{
};

template <int SGSZ> struct GridToSubgrid<1, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap x, Cx3 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = ix + sg[0];
          sx(ix, ib, ic) = x(iix, ib, ic);
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap x, Cx3 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          sx(ix, ib, ic) = x(iix, ib, ic);
        }
      }
    }
  }
};

template <int SGSZ> struct GridToSubgrid<2, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap x, Cx4 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index iy = 0; iy < sx.dimension(1); iy++) {
          Index const iiy = iy + sg[1];
          for (Index ix = 0; ix < sx.dimension(0); ix++) {
            Index const iix = ix + sg[0];
            sx(ix, iy, ib, ic) = x(iix, iiy, ib, ic);
          }
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap x, Cx4 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index iy = 0; iy < sx.dimension(1); iy++) {
          Index const iiy = Wrap(iy + sg[1], x.dimension(1));
          for (Index ix = 0; ix < sx.dimension(0); ix++) {
            Index const iix = Wrap(ix + sg[0], x.dimension(0));
            sx(ix, iy, ib, ic) = x(iix, iiy, ib, ic);
          }
        }
      }
    }
  }
};

template <int SGSZ> struct GridToSubgrid<3, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap x, Cx5 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = iz + sg[2];
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = iy + sg[1];
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = ix + sg[0];
              sx(ix, iy, iz, ib, ic) = x(iix, iiy, iiz, ib, ic);
            }
          }
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap x, Cx5 &sx)
  {
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = Wrap(iz + sg[2], x.dimension(2));
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = Wrap(iy + sg[1], x.dimension(1));
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = Wrap(ix + sg[0], x.dimension(0));
              sx(ix, iy, iz, ib, ic) = x(iix, iiy, iiz, ib, ic);
            }
          }
        }
      }
    }
  }
};

template <int ND, int SGSZ> struct SubgridToGrid
{
};

template <int SGSZ> struct SubgridToGrid<1, SGSZ>
{
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const      iix = ix + corner[0];
          std::scoped_lock lock(m[iix]);
          x(iix, ib, ic) += sx(ix, ib, ic);
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const      iix = Wrap(ix + corner[0], x.dimension(0));
          std::scoped_lock lock(m[iix]);
          x(iix, ib, ic) += sx(ix, ib, ic);
        }
      }
    }
  }
};

template <int SGSZ> struct SubgridToGrid<2, SGSZ>
{
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index iy = 0; iy < SGSZ; iy++) {
          Index const      iiy = iy + corner[1];
          std::scoped_lock lock(m[iiy]);
          for (Index ix = 0; ix < SGSZ; ix++) {
            Index const iix = ix + corner[0];
            x(iix, iiy, ib, ic) += sx(ix, iy, ib, ic);
          }
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index ic = 0; ic < sx.dimension(3); ic++) {
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index iy = 0; iy < SGSZ; iy++) {
          Index const      iiy = Wrap(iy + corner[1], x.dimension(1));
          std::scoped_lock lock(m[iiy]);
          for (Index ix = 0; ix < SGSZ; ix++) {
            Index const iix = Wrap(ix + corner[0], x.dimension(0));
            x(iix, iiy, ib, ic) += sx(ix, iy, ib, ic);
          }
        }
      }
    }
  }
};

template <int SGSZ> struct SubgridToGrid<3, SGSZ>
{
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap sx, Cx5Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index iz = 0; iz < SGSZ; iz++) {
          Index const      iiz = iz + corner[2];
          std::scoped_lock lock(m[iiz]);
          for (Index iy = 0; iy < SGSZ; iy++) {
            Index const iiy = iy + corner[1];
            for (Index ix = 0; ix < SGSZ; ix++) {
              Index const iix = ix + corner[0];
              x(iix, iiy, iiz, ib, ic) += sx(ix, iy, iz, ib, ic);
            }
          }
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap sx, Cx5Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index ic = 0; ic < sx.dimension(4); ic++) {
      for (Index ib = 0; ib < sx.dimension(3); ib++) {
        for (Index iz = 0; iz < SGSZ; iz++) {
          Index const      iiz = Wrap(iz + corner[2], x.dimension(2));
          std::scoped_lock lock(m[iiz]);
          for (Index iy = 0; iy < SGSZ; iy++) {
            Index const iiy = Wrap(iy + corner[1], x.dimension(1));
            for (Index ix = 0; ix < SGSZ; ix++) {
              Index const iix = Wrap(ix + corner[0], x.dimension(0));
              x(iix, iiy, iiz, ib, ic) += sx(ix, iy, iz, ib, ic);
            }
          }
        }
      }
    }
  }
};

} // namespace rl
