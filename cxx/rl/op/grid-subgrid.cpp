#include "grid-subgrid.hpp"

#include <numbers>

namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

namespace rl {

template <>
void GridToSubgrid<1>(Eigen::Array<int16_t, 1, 1> const sg, bool const hasVCC, bool const isVCC, Cx3CMap const &x, Cx3 &sx)
{
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(2));
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ib = 0; ib < sx.dimension(0); ib++) {
        if (hasVCC) {
          if (isVCC) {
            sx(ib, ic, ix) = std::conj(x(ib, cSt + ic, iix)) * inv_sqrt2;
          } else {
            sx(ib, ic, ix) = x(ib, cSt + ic, iix) * inv_sqrt2;
          }
        } else {
          sx(ib, ic, ix) = x(ib, ic, iix);
        }
      }
    }
  }
}

template <>
void GridToSubgrid<2>(Eigen::Array<int16_t, 2, 1> const sg, bool const hasVCC, bool const isVCC, Cx4CMap const &x, Cx4 &sx)
{
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(3));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(2));
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        for (Index ib = 0; ib < sx.dimension(0); ib++) {
          if (hasVCC) {
            if (isVCC) {
              sx(ib, ic, ix, iy) = std::conj(x(ib, cSt + ic, iix, iiy)) * inv_sqrt2;
            } else {
              sx(ib, ic, ix, iy) = x(ib, cSt + ic, iix, iiy) * inv_sqrt2;
            }
          } else {
            sx(ib, ic, ix, iy) = x(ib, ic, iix, iiy);
          }
        }
      }
    }
  }
}

template <>
void GridToSubgrid<3>(Eigen::Array<int16_t, 3, 1> const sg, bool const hasVCC, bool const isVCC, Cx5CMap const &x, Cx5 &sx)
{
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(4));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(2));
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            if (hasVCC) {
              if (isVCC) {
                sx(ib, ic, ix, iy, iz) = std::conj(x(ib, cSt + ic, iix, iiy, iiz)) * inv_sqrt2;
              } else {
                sx(ib, ic, ix, iy, iz) = x(ib, cSt + ic, iix, iiy, iiz) * inv_sqrt2;
              }
            } else {
              sx(ib, ic, ix, iy, iz) = x(ib, ic, iix, iiy, iiz);
            }
          }
        }
      }
    }
  }
}

template <> void GridToSubgrid<1>(Eigen::Array<int16_t, 1, 1> const, bool const, bool const, Cx3CMap const &, Cx3 &);
template <> void GridToSubgrid<2>(Eigen::Array<int16_t, 2, 1> const, bool const, bool const, Cx4CMap const &, Cx4 &);
template <> void GridToSubgrid<3>(Eigen::Array<int16_t, 3, 1> const, bool const, bool const, Cx5CMap const &, Cx5 &);

template <>
void SubgridToGrid<1>(std::vector<std::mutex>          &m,
                      Eigen::Array<int16_t, 1, 1> const sg,
                      bool const                        hasVCC,
                      bool const                        isVCC,
                      Cx3CMap const                    &sx,
                      Cx3Map                           &x)
{
  assert(m.size() == x.dimension(2));
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  float const scale = isVCC ? inv_sqrt2 : 1.f;
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const      iix = Wrap(ix + sg[0], x.dimension(2));
    std::scoped_lock lock(m[iix]);
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ib = 0; ib < sx.dimension(0); ib++) {
        if (hasVCC) {
          if (isVCC) {
            x(ib, cSt + ic, iix) += std::conj(sx(ib, ic, ix)) * inv_sqrt2;
          } else {
            x(ib, cSt + ic, iix) += sx(ib, ic, ix) * inv_sqrt2;
          }
        } else {
          x(ib, ic, iix) += sx(ib, ic, ix);
        }
      }
    }
  }
}

template <>
void SubgridToGrid<2>(std::vector<std::mutex>          &m,
                      Eigen::Array<int16_t, 2, 1> const sg,
                      bool const                        hasVCC,
                      bool const                        isVCC,
                      Cx4CMap const                    &sx,
                      Cx4Map                           &x)
{
  assert(m.size() == x.dimension(3));
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const      iiy = Wrap(iy + sg[1], x.dimension(3));
    std::scoped_lock lock(m[iiy]);
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(2));
      for (Index ic = 0; ic < sx.dimension(1); ic++) {
        for (Index ib = 0; ib < sx.dimension(0); ib++) {
          if (hasVCC) {
            if (isVCC) {
              x(ib, cSt + ic, iix, iiy) += std::conj(sx(ib, ic, ix, iy)) * inv_sqrt2;
            } else {
              x(ib, cSt + ic, iix, iiy) += sx(ib, ic, ix, iy) * inv_sqrt2;
            }
          } else {
            x(ib, ic, iix, iiy) += sx(ib, ic, ix, iy);
          }
        }
      }
    }
  }
}

template <>
void SubgridToGrid<3>(std::vector<std::mutex>          &m,
                      Eigen::Array<int16_t, 3, 1> const sg,
                      bool const                        hasVCC,
                      bool const                        isVCC,
                      Cx5CMap const                    &sx,
                      Cx5Map                           &x)
{
  assert(m.size() == x.dimension(4));
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const      iiz = Wrap(iz + sg[2], x.dimension(4));
    std::scoped_lock lock(m[iiz]);
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(2));
        for (Index ic = 0; ic < sx.dimension(1); ic++) {
          for (Index ib = 0; ib < sx.dimension(0); ib++) {
            if (hasVCC) {
              if (isVCC) {
                x(ib, cSt + ic, iix, iiy, iiz) += std::conj(sx(ib, ic, ix, iy, iz)) * inv_sqrt2;
              } else {
                x(ib, cSt + ic, iix, iiy, iiz) += sx(ib, ic, ix, iy, iz) * inv_sqrt2;
              }
            } else {
              x(ib, ic, iix, iiy, iiz) += sx(ib, ic, ix, iy, iz);
            }
          }
        }
      }
    }
  }
}

template <>
void SubgridToGrid<1>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 1, 1> const, bool const, bool const, Cx3CMap const &, Cx3Map &);
template <>
void SubgridToGrid<2>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 2, 1> const, bool const, bool const, Cx4CMap const &, Cx4Map &);
template <>
void SubgridToGrid<3>(
  std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const, bool const, bool const, Cx5CMap const &, Cx5Map &);

} // namespace rl
