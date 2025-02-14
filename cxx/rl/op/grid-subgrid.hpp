#pragma once

#include "../types.hpp"
#include <mutex>

namespace rl {

template <int ND, int SGSZ> struct GridToSubgrid
{
};

template <int SGSZ> struct GridToSubgrid<1, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap const x, Cx3 &sx)
  {
    for (Index ix = 0; ix < SGSZ; ix++) {
      Index const iix = ix + sg[0];
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          sx(ix, ic, ib) = x(iix, ic, ib);
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap const x, Cx3 &sx)
  {
    for (Index ix = 0; ix < SGSZ; ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(0));
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          sx(ix, ic, ib) = x(iix, ic, ib);
        }
      }
    }
  }
};

template <int SGSZ> struct GridToSubgrid<2, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap const x, Cx4 &sx)
  {
    for (Index iy = 0; iy < SGSZ; iy++) {
      Index const iiy = iy + sg[1];
      for (Index ix = 0; ix < SGSZ; ix++) {
        Index const iix = ix + sg[0];
        for (Index ib = 0; ib < sx.dimension(3); ib++) {
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            sx(ix, iy, ic, ib) = x(iix, iiy, ic, ib);
          }
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap const x, Cx4 &sx)
  {
    for (Index iy = 0; iy < SGSZ; iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(1));
      for (Index ix = 0; ix < SGSZ; ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(0));
        for (Index ib = 0; ib < sx.dimension(3); ib++) {
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            sx(ix, iy, ic, ib) = x(iix, iiy, ic, ib);
          }
        }
      }
    }
  }
};

template <int SGSZ> struct GridToSubgrid<3, SGSZ>
{
  inline static void FastCopy(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap const x, Cx5 &sx)
  {
    for (Index iz = 0; iz < SGSZ; iz++) {
      Index const iiz = iz + sg[2];
      for (Index iy = 0; iy < SGSZ; iy++) {
        Index const iiy = iy + sg[1];
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const iix = ix + sg[0];
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
            }
          }
        }
      }
    }
  }

  inline static void SlowCopy(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap const x, Cx5 &sx)
  {

    for (Index iz = 0; iz < SGSZ; iz++) {
      Index const iiz = Wrap(iz + sg[2], x.dimension(2));
      for (Index iy = 0; iy < SGSZ; iy++) {
        Index const iiy = Wrap(iy + sg[1], x.dimension(1));
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
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
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap const sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ix = 0; ix < SGSZ; ix++) {
      Index const      iix = ix + corner[0];
      std::scoped_lock lock(m[iix]);
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          x(iix, ic, ib) += sx(ix, ic, ib);
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const corner, Cx3CMap const sx, Cx3Map x)
  {
    assert(m.size() == x.dimension(0));
    for (Index ix = 0; ix < SGSZ; ix++) {
      Index const      iix = Wrap(ix + corner[0], x.dimension(0));
      std::scoped_lock lock(m[iix]);
      for (Index ib = 0; ib < sx.dimension(2); ib++) {
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          x(iix, ic, ib) += sx(ix, ic, ib);
        }
      }
    }
  }
};

template <int SGSZ> struct SubgridToGrid<2, SGSZ>
{
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap const sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index iy = 0; iy < SGSZ; iy++) {
      Index const      iiy = iy + corner[1];
      std::scoped_lock lock(m[iiy]);
      for (Index ix = 0; ix < SGSZ; ix++) {
        Index const iix = ix + corner[0];
        for (Index ib = 0; ib < sx.dimension(3); ib++) {
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            x(iix, iiy, ic, ib) += sx(ix, iy, ic, ib);
          }
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const corner, Cx4CMap const sx, Cx4Map x)
  {
    assert(m.size() == x.dimension(1));
    for (Index iy = 0; iy < SGSZ; iy++) {
      Index const      iiy = Wrap(iy + corner[1], x.dimension(1));
      std::scoped_lock lock(m[iiy]);
      for (Index ix = 0; ix < SGSZ; ix++) {
        Index const iix = Wrap(ix + corner[0], x.dimension(0));
        for (Index ib = 0; ib < sx.dimension(3); ib++) {
          for (Index ic = 0; ic < sx.dimension(2); ic++) {
            x(iix, iiy, ic, ib) += sx(ix, iy, ic, ib);
          }
        }
      }
    }
  }
};

template <int SGSZ> struct SubgridToGrid<3, SGSZ>
{
  inline static void FastCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap const sx, Cx5Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index iz = 0; iz < SGSZ; iz++) {
      Index const      iiz = iz + corner[2];
      std::scoped_lock lock(m[iiz]);
      for (Index iy = 0; iy < SGSZ; iy++) {
        Index const iiy = iy + corner[1];
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const iix = ix + corner[0];
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
            }
          }
        }
      }
    }
  }

  inline static void SlowCopy(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap const sx, Cx5Map x)
  {
    assert(m.size() == x.dimension(2));
    for (Index iz = 0; iz < SGSZ; iz++) {
      Index const      iiz = Wrap(iz + corner[2], x.dimension(2));
      std::scoped_lock lock(m[iiz]);
      for (Index iy = 0; iy < SGSZ; iy++) {
        Index const iiy = Wrap(iy + corner[1], x.dimension(1));
        for (Index ix = 0; ix < SGSZ; ix++) {
          Index const iix = Wrap(ix + corner[0], x.dimension(0));
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
            }
          }
        }
      }
    }
  }
};

} // namespace rl
