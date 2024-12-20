#include "grid-subgrid.hpp"

#include "fmt/ranges.h"
#include <numbers>
namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

namespace rl {

template <>
void GridToSubgrid<1>(Eigen::Array<int16_t, 1, 1> const sg, bool const hasVCC, bool const isVCC, Cx3CMap const &x, Cx3 &sx)
{
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  for (Index ib = 0; ib < sx.dimension(2); ib++) {
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(0));
        if (hasVCC) {
          if (isVCC) {
            sx(ix, ic, ib) = std::conj(x(iix, cSt + ic, ib)) * inv_sqrt2;
          } else {
            sx(ix, ic, ib) = x(iix, cSt + ic, ib) * inv_sqrt2;
          }
        } else {
          sx(ix, ic, ib) = x(iix, ic, ib);
        }
      }
    }
  }
}

template <>
void GridToSubgrid<2>(Eigen::Array<int16_t, 2, 1> const sg, bool const hasVCC, bool const isVCC, Cx4CMap const &x, Cx4 &sx)
{
  Index const cSt = isVCC ? sx.dimension(2) : 0;
  for (Index ib = 0; ib < sx.dimension(3); ib++) {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const iiy = Wrap(iy + sg[1], x.dimension(1));
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          if (hasVCC) {
            if (isVCC) {
              sx(ix, iy, ic, ib) = std::conj(x(iix, iiy, cSt + ic, ib)) * inv_sqrt2;
            } else {
              sx(ix, iy, ic, ib) = x(iix, iiy, cSt + ic, ib) * inv_sqrt2;
            }
          } else {
            sx(ix, iy, ic, ib) = x(iix, iiy, ic, ib);
          }
        }
      }
    }
  }
}

template <>
void GridToSubgrid<3>(Eigen::Array<int16_t, 3, 1> const sg, bool const hasVCC, bool const isVCC, Cx5CMap const &x, Cx5 &sx)
{

  bool inBounds = true;
  for (Index ii = 0; ii < 3; ii++) {
    if (sg[ii] < 0 || (sg[ii] + sx.dimension(ii) >= x.dimension(ii))) { inBounds = false; }
  }

  if (inBounds) {
    // fmt::print(stderr, "Fast path sg {} sx {} x {}\n", sg, sx.dimensions(), x.dimensions());
    Index const cSt = isVCC ? sx.dimension(3) : 0;
    for (Index ib = 0; ib < sx.dimension(4); ib++) {
      for (Index ic = 0; ic < sx.dimension(3); ic++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = iz + sg[2];
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = iy + sg[1];
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = ix + sg[0];
              // if (hasVCC) {
              //   if (isVCC) {
              //     sx(ix, iy, iz, ic, ib) = std::conj(x(iix, iiy, iiz, cSt + ic, ib)) * inv_sqrt2;
              //   } else {
              //     sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, cSt + ic, ib) * inv_sqrt2;
              //   }
              // } else {
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
              // }
            }
          }
        }
      }
    }
  } else {
    // fmt::print(stderr, "Slow path sg {} sx {} x {}\n", sg, sx.dimensions(), x.dimensions());
    Index const cSt = isVCC ? sx.dimension(3) : 0;
    for (Index ib = 0; ib < sx.dimension(4); ib++) {
      for (Index ic = 0; ic < sx.dimension(3); ic++) {
        for (Index iz = 0; iz < sx.dimension(2); iz++) {
          Index const iiz = Wrap(iz + sg[2], x.dimension(2));
          for (Index iy = 0; iy < sx.dimension(1); iy++) {
            Index const iiy = Wrap(iy + sg[1], x.dimension(1));
            for (Index ix = 0; ix < sx.dimension(0); ix++) {
              Index const iix = Wrap(ix + sg[0], x.dimension(2));
              // if (hasVCC) {
              //   if (isVCC) {
              //     sx(ix, iy, iz, ic, ib) = std::conj(x(iix, iiy, iiz, cSt + ic, ib)) * inv_sqrt2;
              //   } else {
              //     sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, cSt + ic, ib) * inv_sqrt2;
              //   }
              // } else {
              sx(ix, iy, iz, ic, ib) = x(iix, iiy, iiz, ic, ib);
              // }
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
  assert(m.size() == x.dimension(0));
  Index const cSt = isVCC ? sx.dimension(1) : 0;
  float const scale = isVCC ? inv_sqrt2 : 1.f;
  for (Index ib = 0; ib < sx.dimension(2); ib++) {
    for (Index ic = 0; ic < sx.dimension(1); ic++) {
      for (Index ix = 0; ix < sx.dimension(0); ix++) {
        Index const      iix = Wrap(ix + sg[0], x.dimension(0));
        std::scoped_lock lock(m[iix]);
        if (hasVCC) {
          if (isVCC) {
            x(iix, cSt + ic, ib) += std::conj(sx(ix, ic, ib)) * inv_sqrt2;
          } else {
            x(iix, cSt + ic, ib) += sx(ix, ic, ib) * inv_sqrt2;
          }
        } else {
          x(iix, ic, ib) += sx(ix, ic, ib);
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
  assert(m.size() == x.dimension(1));
  Index const cSt = isVCC ? sx.dimension(2) : 0;
  for (Index ib = 0; ib < sx.dimension(3); ib++) {
    for (Index ic = 0; ic < sx.dimension(2); ic++) {
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const      iiy = Wrap(iy + sg[1], x.dimension(1));
        std::scoped_lock lock(m[iiy]);
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          if (hasVCC) {
            if (isVCC) {
              x(iix, iiy, cSt + ic, ib) += std::conj(sx(ix, iy, ic, ib)) * inv_sqrt2;
            } else {
              x(iix, iiy, cSt + ic, ib) += sx(ix, iy, ic, ib) * inv_sqrt2;
            }
          } else {
            x(iix, iiy, ic, ib) += sx(ix, iy, ic, ib);
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
  assert(m.size() == x.dimension(2));
  Index const cSt = isVCC ? sx.dimension(3) : 0;

  bool inBounds = true;
  for (Index ii = 0; ii < 3; ii++) {
    if (sg[ii] < 0 || (sg[ii] + sx.dimension(ii) >= x.dimension(ii))) { inBounds = false; }
  }

  if (inBounds) {
    for (Index iz = 0; iz < sx.dimension(2); iz++) {
      Index const      iiz = iz + sg[2];
      std::scoped_lock lock(m[iiz]);
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const iiy = iy + sg[1];
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = ix + sg[0];
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              if (hasVCC) {
                if (isVCC) {
                  x(iix, iiy, iiz, cSt + ic, ib) += std::conj(sx(ix, iy, iz, ic, ib)) * inv_sqrt2;
                } else {
                  x(iix, iiy, iiz, cSt + ic, ib) += sx(ix, iy, iz, ic, ib) * inv_sqrt2;
                }
              } else {
                x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
              }
            }
          }
        }
      }
    }
  } else {
    for (Index iz = 0; iz < sx.dimension(2); iz++) {
      Index const      iiz = Wrap(iz + sg[2], x.dimension(2));
      std::scoped_lock lock(m[iiz]);
      for (Index iy = 0; iy < sx.dimension(1); iy++) {
        Index const iiy = Wrap(iy + sg[1], x.dimension(1));
        for (Index ix = 0; ix < sx.dimension(0); ix++) {
          Index const iix = Wrap(ix + sg[0], x.dimension(0));
          for (Index ib = 0; ib < sx.dimension(4); ib++) {
            for (Index ic = 0; ic < sx.dimension(3); ic++) {
              if (hasVCC) {
                if (isVCC) {
                  x(iix, iiy, iiz, cSt + ic, ib) += std::conj(sx(ix, iy, iz, ic, ib)) * inv_sqrt2;
                } else {
                  x(iix, iiy, iiz, cSt + ic, ib) += sx(ix, iy, iz, ic, ib) * inv_sqrt2;
                }
              } else {
                x(iix, iiy, iiz, ic, ib) += sx(ix, iy, iz, ic, ib);
              }
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
