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

template <> template <> void Subgrid<1, false>::subgridToGrid<false>(Cx3CMap const &sx, Cx3Map &x) const
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, ib, iix) += sx(ic, ib, ix);
      }
    }
  }
}

template <> template <> void Subgrid<2, false>::subgridToGrid<false>(Cx4CMap const &sx, Cx4Map &x) const
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(3));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, ib, iix, iiy) += sx(ic, ib, ix, iy);
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, false>::subgridToGrid<false>(Cx5CMap const &sx, Cx5Map &x) const
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(4));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(2));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            x(ic, ib, iix, iiy, iiz) += sx(ic, ib, ix, iy, iz);
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

template <> template <> void Subgrid<1, false>::subgridToGrid<true>(Cx3CMap const &sx, Cx3Map &x) const
{
  /* Never called, no-op */
}

template <> template <> void Subgrid<2, false>::subgridToGrid<true>(Cx4CMap const &sx, Cx4Map &x) const
{
  /* Never called, no-op */
}

template <> template <> void Subgrid<3, false>::subgridToGrid<true>(Cx5CMap const &sx, Cx5Map &x) const
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

template <> template <> void Subgrid<1, true>::subgridToGrid<false>(Cx3CMap const &sx, Cx4Map &x) const
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, 0, ib, iix) = sx(ic, ib, ix) * inv_sqrt2;
      }
    }
  }
}

template <> template <> void Subgrid<2, true>::subgridToGrid<false>(Cx4CMap const &sx, Cx5Map &x) const
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, 0, ib, iix, iiy) = sx(ic, ib, ix, iy) * inv_sqrt2;
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, true>::subgridToGrid<false>(Cx5CMap const &sx, Cx6Map &x) const
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            x(ic, 0, ib, iix, iiy, iiz) = sx(ic, ib, ix, iy, iz) * inv_sqrt2;
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

template <> template <> void Subgrid<1, true>::subgridToGrid<true>(Cx3CMap const &sx, Cx4Map &x) const
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, 1, ib, iix) = std::conj(sx(ic, ib, ix)) * inv_sqrt2;
      }
    }
  }
}

template <> template <> void Subgrid<2, true>::subgridToGrid<true>(Cx4CMap const &sx, Cx5Map &x) const
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, 1, ib, iix, iiy) = std::conj(sx(ic, ib, ix, iy)) * inv_sqrt2;
        }
      }
    }
  }
}

template <> template <> void Subgrid<3, true>::subgridToGrid<true>(Cx5CMap const &sx, Cx6Map &x) const
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + minCorner[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + minCorner[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + minCorner[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            x(ic, 1, ib, iix, iiy, iiz) = std::conj(sx(ic, ib, ix, iy, iz)) * inv_sqrt2;
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
