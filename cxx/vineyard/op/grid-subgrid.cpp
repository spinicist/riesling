#include "grid-subgrid.hpp"

#include <numbers>

namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

namespace rl {

/*****************************************************************************************************
 * No VCC at all
 ****************************************************************************************************/
template <> void GridToSubgrid<1, false, false>(Sz1 const sg, Cx3CMap const &x, Cx3 &sx)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(2));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = x(ic, ib, iix);
      }
    }
  }
}

template <> void GridToSubgrid<2, false, false>(Sz2 const sg, Cx4CMap const &x, Cx4 &sx)
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(3));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(2));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = x(ic, ib, iix, iiy);
        }
      }
    }
  }
}

template <> void GridToSubgrid<3, false, false>(Sz3 const sg, Cx5CMap const &x, Cx5 &sx)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(4));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(2));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = x(ic, ib, iix, iiy, iiz);
          }
        }
      }
    }
  }
}

template <> void SubgridToGrid<1, false, false>(Sz1 const sg, Cx3CMap const &sx, Cx3Map &x)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(2));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, ib, iix) += sx(ic, ib, ix);
      }
    }
  }
}

template <> void SubgridToGrid<2, false, false>(Sz2 const sg, Cx4CMap const &sx, Cx4Map &x)
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(3));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(2));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, ib, iix, iiy) += sx(ic, ib, ix, iy);
        }
      }
    }
  }
}

template <> void SubgridToGrid<3, false, false>(Sz3 const sg, Cx5CMap const &sx, Cx5Map &x)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(4));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(3));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(2));
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
 * Has VCC but is not VCC
 ****************************************************************************************************/
template <> void GridToSubgrid<1, true, false>(Sz1 const sg, Cx4CMap const &x, Cx3 &sx)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = x(ic, 0, ib, iix) * inv_sqrt2;
      }
    }
  }
}

template <> void GridToSubgrid<2, true, false>(Sz2 const sg, Cx5CMap const &x, Cx4 &sx)
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = x(ic, 0, ib, iix, iiy) * inv_sqrt2;
        }
      }
    }
  }
}

template <> void GridToSubgrid<3, true, false>(Sz3 const sg, Cx6CMap const &x, Cx5 &sx)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = x(ic, 0, ib, iix, iiy, iiz) * inv_sqrt2;
          }
        }
      }
    }
  }
}

template <> void SubgridToGrid<1, true, false>(Sz1 const sg, Cx3CMap const &sx, Cx4Map &x)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, 0, ib, iix) += sx(ic, ib, ix) * inv_sqrt2;
      }
    }
  }
}

template <> void SubgridToGrid<2, true, false>(Sz2 const sg, Cx4CMap const &sx, Cx5Map &x) 
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, 0, ib, iix, iiy) += sx(ic, ib, ix, iy) * inv_sqrt2;
        }
      }
    }
  }
}

template <> void SubgridToGrid<3, true, false>(Sz3 const sg, Cx5CMap const &sx, Cx6Map &x)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            x(ic, 0, ib, iix, iiy, iiz) += sx(ic, ib, ix, iy, iz) * inv_sqrt2;
          }
        }
      }
    }
  }
}
/*****************************************************************************************************
 * Has VCC and is VCC
 ****************************************************************************************************/
template <> void GridToSubgrid<1, true, true>(Sz1 const sg, Cx4CMap const &x, Cx3 &sx)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        sx(ic, ib, ix) = std::conj(x(ic, 1, ib, iix)) * inv_sqrt2;
      }
    }
  }
}

template <> void GridToSubgrid<2, true, true>(Sz2 const sg, Cx5CMap const &x, Cx4 &sx)
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          sx(ic, ib, ix, iy) = std::conj(x(ic, 1, ib, iix, iiy)) * inv_sqrt2;
        }
      }
    }
  }
}

template <> void GridToSubgrid<3, true, true>(Sz3 const sg, Cx6CMap const &x, Cx5 &sx)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            sx(ic, ib, ix, iy, iz) = std::conj(x(ic, 1, ib, iix, iiy, iiz)) * inv_sqrt2;
          }
        }
      }
    }
  }
}

template <> void SubgridToGrid<1, true, true>(Sz1 const sg, Cx3CMap const &sx, Cx4Map &x)
{
  for (Index ix = 0; ix < sx.dimension(2); ix++) {
    Index const iix = Wrap(ix + sg[0], x.dimension(3));
    for (Index ib = 0; ib < sx.dimension(1); ib++) {
      for (Index ic = 0; ic < sx.dimension(0); ic++) {
        x(ic, 1, ib, iix) += std::conj(sx(ic, ib, ix)) * inv_sqrt2;
      }
    }
  }
}

template <> void SubgridToGrid<2, true, true>(Sz2 const sg, Cx4CMap const &sx, Cx5Map &x)
{
  for (Index iy = 0; iy < sx.dimension(3); iy++) {
    Index const iiy = Wrap(iy + sg[1], x.dimension(4));
    for (Index ix = 0; ix < sx.dimension(2); ix++) {
      Index const iix = Wrap(ix + sg[0], x.dimension(3));
      for (Index ib = 0; ib < sx.dimension(1); ib++) {
        for (Index ic = 0; ic < sx.dimension(0); ic++) {
          x(ic, 1, ib, iix, iiy) += std::conj(sx(ic, ib, ix, iy)) * inv_sqrt2;
        }
      }
    }
  }
}

template <> void SubgridToGrid<3, true, true>(Sz3 const sg, Cx5CMap const &sx, Cx6Map &x)
{
  for (Index iz = 0; iz < sx.dimension(4); iz++) {
    Index const iiz = Wrap(iz + sg[2], x.dimension(5));
    for (Index iy = 0; iy < sx.dimension(3); iy++) {
      Index const iiy = Wrap(iy + sg[1], x.dimension(4));
      for (Index ix = 0; ix < sx.dimension(2); ix++) {
        Index const iix = Wrap(ix + sg[0], x.dimension(3));
        for (Index ib = 0; ib < sx.dimension(1); ib++) {
          for (Index ic = 0; ic < sx.dimension(0); ic++) {
            x(ic, 1, ib, iix, iiy, iiz) += std::conj(sx(ic, ib, ix, iy, iz)) * inv_sqrt2;
          }
        }
      }
    }
  }
}

template <> void GridToSubgrid<1, false, false>(Sz1 const, Cx3CMap const &, Cx3 &);
template <> void GridToSubgrid<2, false, false>(Sz2 const, Cx4CMap const &, Cx4 &);
template <> void GridToSubgrid<3, false, false>(Sz3 const, Cx5CMap const &, Cx5 &);

template <> void SubgridToGrid<1, false, false>(Sz1 const, Cx3CMap const &, Cx3Map &);
template <> void SubgridToGrid<2, false, false>(Sz2 const, Cx4CMap const &, Cx4Map &);
template <> void SubgridToGrid<3, false, false>(Sz3 const, Cx5CMap const &, Cx5Map &);

template <> void GridToSubgrid<1, true, false>(Sz1 const, Cx4CMap const &, Cx3 &);
template <> void GridToSubgrid<2, true, false>(Sz2 const, Cx5CMap const &, Cx4 &);
template <> void GridToSubgrid<3, true, false>(Sz3 const, Cx6CMap const &, Cx5 &);

template <> void SubgridToGrid<1, true, false>(Sz1 const, Cx3CMap const &, Cx4Map &);
template <> void SubgridToGrid<2, true, false>(Sz2 const, Cx4CMap const &, Cx5Map &);
template <> void SubgridToGrid<3, true, false>(Sz3 const, Cx5CMap const &, Cx6Map &);

template <> void GridToSubgrid<1, true, true>(Sz1 const, Cx4CMap const &, Cx3 &);
template <> void GridToSubgrid<2, true, true>(Sz2 const, Cx5CMap const &, Cx4 &);
template <> void GridToSubgrid<3, true, true>(Sz3 const, Cx6CMap const &, Cx5 &);

template <> void SubgridToGrid<1, true, true>(Sz1 const, Cx3CMap const &, Cx4Map &);
template <> void SubgridToGrid<2, true, true>(Sz2 const, Cx4CMap const &, Cx5Map &);
template <> void SubgridToGrid<3, true, true>(Sz3 const, Cx5CMap const &, Cx6Map &);


} // namespace rl
