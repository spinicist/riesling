#include "grid-subgrid.hpp"

#include "fmt/ranges.h"

namespace rl {

template <> void GridToSubgrid<1>(Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap const &x, Cx3 &sx)
{
  for (Index ib = 0; ib < sx.dimension(2); ib++) {
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(0));
        sx(ix, ic, ib) = x(iix, ic, ib);
      }
    }
  }
}

template <> void GridToSubgrid<2>(Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap const &x, Cx4 &sx)
{
  for (Index ib = 0; ib < sx.dimension(3); ib++) {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const iiy = Wrap(iy + sg[1], x.dimension(1));
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          sx(ix, iy, ic, ib) = x(iix, iiy, ic, ib);
        }
      }
    }
  }
}

template <> void GridToSubgrid<3>(Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap const &x, Cx5 &sx)
{
  bool inBounds = true;
  for (Index ii = 0; ii < 3; ii++) {
    if (sg[ii] < 0 || (sg[ii] + sx.dimension(ii) >= x.dimension(ii))) { inBounds = false; }
  }

  if (inBounds) {
    fmt::print(stderr, "Fast path sg {} sx {} x {}\n", sg, sx.dimensions(), x.dimensions());
    for (Index ib = 0; ib < sx.dimension(4); ib++) {
      for (Index ic = 0; ic < sx.dimension(3); ic++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = iz + sg[2];
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = iy + sg[1];
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = ix + sg[0];
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
            }
          }
        }
      }
    }
  } else {
    fmt::print(stderr, "Slow path sg {} sx {} x {}\n", sg, sx.dimensions(), x.dimensions());
    for (Index ib = 0; ib < sx.dimension(4); ib++) {
      for (Index ic = 0; ic < sx.dimension(3); ic++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = Wrap(iz + sg[2], x.dimension(2));
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = Wrap(iy + sg[1], x.dimension(1));
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = Wrap(ix + sg[0], x.dimension(0));
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
            }
          }
        }
      }
    }
  }
}

template <> void GridToSubgrid<1>(Eigen::Array<int16_t, 1, 1> const, Cx3CMap const &, Cx3 &);
template <> void GridToSubgrid<2>(Eigen::Array<int16_t, 2, 1> const, Cx4CMap const &, Cx4 &);
template <> void GridToSubgrid<3>(Eigen::Array<int16_t, 3, 1> const, Cx5CMap const &, Cx5 &);

template <>
void SubgridToGrid<1>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const sg, Cx3CMap const &sx, Cx3Map &x)
{
  assert(m.size() == x.dimension(0));
  for (Index ib = 0; ib < sx.dimension(2); ib++) {
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        Index const      iix = Wrap(ix + sg[0], x.dimension(0));
        std::scoped_lock lock(m[iix]);
        x(iix, ic, ib) += sx(ix, ic, ib);
      }
    }
  }
}

template <>
void SubgridToGrid<2>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const sg, Cx4CMap const &sx, Cx4Map &x)
{
  assert(m.size() == x.dimension(1));
  for (Index ib = 0; ib < sx.dimension(3); ib++) {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const      iiy = Wrap(iy + sg[1], x.dimension(1));
        std::scoped_lock lock(m[iiy]);
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          x(iix, iiy, ic, ib) += sx(ix, iy, ic, ib);
        }
      }
    }
  }
}

template <>
void SubgridToGrid<3>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const sg, Cx5CMap const &sx, Cx5Map &x)
{
  assert(m.size() == x.dimension(2));

  bool inBounds = true;
  for (Index ii = 0; ii < 3; ii++) {
    if (sg[ii] < 0 || (sg[ii] + sx.dimension(ii) >= x.dimension(ii))) { inBounds = false; }
  }

  if (inBounds) {
    for (Index iz = 0; iz < sx.dimension(2); iz++) {
      Index const      iiz = iz + sg[2];
      std::scoped_lock lock(m[iiz]);
      for (Index ib = 0; ib < sx.dimension(4); ib++) {
        for (Index ic = 0; ic < sx.dimension(3); ic++) {

          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = iy + sg[1];
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = ix + sg[0];
              x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
            }
          }
        }
      }
    }
  } else {
    for (Index iz = 0; iz < sx.dimension(2); iz++) {
      Index const      iiz = Wrap(iz + sg[2], x.dimension(2));
      std::scoped_lock lock(m[iiz]);
      for (Index ib = 0; ib < sx.dimension(4); ib++) {
        for (Index ic = 0; ic < sx.dimension(3); ic++) {
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = Wrap(iy + sg[1], x.dimension(1));
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = Wrap(ix + sg[0], x.dimension(0));
              x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
            }
          }
        }
      }
    }
  }
}

template <> void SubgridToGrid<1>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const, Cx3CMap const &, Cx3Map &);
template <> void SubgridToGrid<2>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const, Cx4CMap const &, Cx4Map &);
template <> void SubgridToGrid<3>(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const, Cx5CMap const &, Cx5Map &);

} // namespace rl
