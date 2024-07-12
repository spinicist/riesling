#include "grid-subgrid.hpp"

#include <numbers>

namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

namespace rl {

template <int ND, bool VCC> auto Subgrid<ND, VCC>::empty() const -> bool { return indices.empty(); }

template <int ND, bool VCC> auto Subgrid<ND, VCC>::count() const -> Index { return indices.size(); }

template <int ND, bool VCC> auto Subgrid<ND, VCC>::size() const -> Sz<ND>
{
  Sz<ND> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

/*****************************************************************************************************
 * No VCC at all
 ****************************************************************************************************/
template <> template <> void Subgrid<1, false>::gridToSubgrid<false>(Cx3CMap const &x, Cx3 &sx) const
{
  sx.setZero();
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = x(ic, ib, iix);
      }
    }
  }
}

template <> template <> void Subgrid<2, false>::gridToSubgrid<false>(Cx4CMap const &x, Cx4 &sx) const
{
  sx.setZero();
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(3));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = x(ic, ib, iix, iiy);
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, false>::gridToSubgrid<false>(Cx5CMap const &x, Cx5 &sx) const
{
  sx.setZero();
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(4));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = x(ic, ib, iix, iiy, iiz);
          }
        }
      }
    }
  }
}

/*****************************************************************************************************
 * NO VCC but is VCC - compiler not happy if these don't exist
 ****************************************************************************************************/
template <> template <> void Subgrid<1, false>::gridToSubgrid<true>(Cx3CMap const &x, Cx3 &sx) const
{
  /* Never called, no-op */
}

template <> template <> void Subgrid<2, false>::gridToSubgrid<true>(Cx4CMap const &x, Cx4 &sx) const
{
  /* Never called, no-op */
}

template <> template <> void Subgrid<3, false>::gridToSubgrid<true>(Cx5CMap const &x, Cx5 &sx) const
{
  /* Never called, no-op */
}

/*****************************************************************************************************
 * Has VCC but is not VCC
 ****************************************************************************************************/
template <> template <> void Subgrid<1, true>::gridToSubgrid<false>(Cx4CMap const &x, Cx3 &sx) const
{
  sx.setZero();
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = x(ic, 0, ib, iix) * inv_sqrt2;
      }
    }
  }
}

template <> template <> void Subgrid<2, true>::gridToSubgrid<false>(Cx5CMap const &x, Cx4 &sx) const
{
  sx.setZero();
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = x(ic, 0, ib, iix, iiy) * inv_sqrt2;
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, true>::gridToSubgrid<false>(Cx6CMap const &x, Cx5 &sx) const
{
  sx.setZero();
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = x(ic, 0, ib, iix, iiy, iiz) * inv_sqrt2;
          }
        }
      }
    }
  }
}

/*****************************************************************************************************
 * Has VCC and is VCC
 ****************************************************************************************************/
template <> template <> void Subgrid<1, true>::gridToSubgrid<true>(Cx4CMap const &x, Cx3 &sx) const
{
  sx.setZero();
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = std::conj(x(ic, 1, ib, iix)) * inv_sqrt2;
      }
    }
  }
}

template <> template <> void Subgrid<2, true>::gridToSubgrid<true>(Cx5CMap const &x, Cx4 &sx) const
{
  sx.setZero();
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = std::conj(x(ic, 1, ib, iix, iiy)) * inv_sqrt2;
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, true>::gridToSubgrid<true>(Cx6CMap const &x, Cx5 &sx) const
{
  sx.setZero();
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = std::conj(x(ic, 1, ib, iix, iiy, iiz)) * inv_sqrt2;
          }
        }
      }
    }
  }
}

template struct Subgrid<1, false>;
template struct Subgrid<2, false>;
template struct Subgrid<3, false>;

template struct Subgrid<1, true>;
template struct Subgrid<2, true>;
template struct Subgrid<3, true>;

} // namespace rl
