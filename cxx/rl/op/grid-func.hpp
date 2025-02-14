#pragma once

#include "../basis/basis.hpp"
#include "../types.hpp"

namespace rl {

template <int ND, int SGSZ, int KW>
inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd) -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * SGSZ) - (KW / 2);
}

template <int ND, int SGSZ> inline auto InBounds(Eigen::Array<int16_t, ND, 1> const corner, Sz<ND> const gSz)
{
  bool inBounds = true;
  for (Index ii = 0; ii < ND; ii++) {
    if (corner[ii] < 0 || (corner[ii] + SGSZ >= gSz[ii])) { inBounds = false; }
  }
  return inBounds;
}

template <int ND, int SGSZ>
inline auto InBounds(Eigen::Array<int16_t, ND, 1> const corner, Sz<ND> const gSz, Sz<ND> const kernSz)
{
  bool inBounds = true;
  for (Index ii = 0; ii < ND; ii++) {
    if (corner[ii] - kernSz[ii] / 2 < 0 || (corner[ii] + SGSZ + kernSz[ii] / 2 >= gSz[ii])) { inBounds = false; }
  }
  return inBounds;
}

template <int ND, int FW> struct GFunc
{
};

template <int FW> struct GFunc<1, FW>
{
  using KT = FixedTensor<float, 1, FW>;

  inline static void
  Scatter(Eigen::Array<int16_t, 1, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx3CMap y, Cx3Map sg)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index ix = 0; ix < FW; ix++) {
        Index const iix = ix + c[0] - FW / 2;
        sg(iix, ic, 0) += y(ic, sample, trace) * k(ix);
      }
    }
  }

  inline static void
  Gather(Eigen::Array<int16_t, 1, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx3CMap sg, Cx3Map y)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index ix = 0; ix < FW; ix++) {
        Index const iix = ix + c[0] - FW / 2;
        y(ic, sample, trace) += sg(iix, ic, 0) * k(ix);
      }
    }
  }

  inline static void Scatter(Basis::CPtr                       basis,
                             Eigen::Array<int16_t, 1, 1> const c,
                             int16_t const                     sample,
                             int32_t const                     trace,
                             KT const                         &k,
                             Cx3CMap                           y,
                             Cx3Map                            sg)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index ix = 0; ix < FW; ix++) {
          Index const iix = ix + c[0] - FW / 2;
          sg(iix, ic, ib) += y(ic, sample, trace) * k(ix) * b;
        }
      }
    }
  }

  inline static void Gather(Basis::CPtr                       basis,
                            Eigen::Array<int16_t, 1, 1> const c,
                            int16_t const                     sample,
                            int32_t const                     trace,
                            KT const                         &k,
                            Cx3CMap                           sg,
                            Cx3Map                            y)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index ix = 0; ix < FW; ix++) {
          Index const iix = ix + c[0] - FW / 2;
          y(ic, sample, trace) += sg(iix, ic, ib) * k(ix) * b;
        }
      }
    }
  }
};

template <int FW> struct GFunc<2, FW>
{
  using KT = FixedTensor<float, 2, FW>;

  inline static void
  Scatter(Eigen::Array<int16_t, 2, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx3CMap y, Cx4Map sg)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index iy = 0; iy < FW; iy++) {
        Index const iiy = iy + c[1] - FW / 2;
        for (Index ix = 0; ix < FW; ix++) {
          Index const iix = ix + c[0] - FW / 2;
          sg(iix, iiy, ic, 0) += y(ic, sample, trace) * k(ix, iy);
        }
      }
    }
  }

  inline static void
  Gather(Eigen::Array<int16_t, 2, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx4CMap sg, Cx3Map y)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index iy = 0; iy < FW; iy++) {
        Index const iiy = iy + c[1] - FW / 2;
        for (Index ix = 0; ix < FW; ix++) {
          Index const iix = ix + c[0] - FW / 2;
          y(ic, sample, trace) += sg(iix, iiy, ic, 0) * k(ix, iy);
        }
      }
    }
  }

  inline static void Scatter(Basis::CPtr                       basis,
                             Eigen::Array<int16_t, 2, 1> const c,
                             int16_t const                     sample,
                             int32_t const                     trace,
                             KT const                         &k,
                             Cx3CMap                           y,
                             Cx4Map                            sg)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index iy = 0; iy < FW; iy++) {
          Index const iiy = iy + c[1] - FW / 2;
          for (Index ix = 0; ix < FW; ix++) {
            Index const iix = ix + c[0] - FW / 2;
            sg(iix, iiy, ic, ib) += y(ic, sample, trace) * k(ix, iy) * b;
          }
        }
      }
    }
  }

  inline static void Gather(Basis::CPtr                       basis,
                            Eigen::Array<int16_t, 2, 1> const c,
                            int16_t const                     sample,
                            int32_t const                     trace,
                            KT const                         &k,
                            Cx4CMap                           sg,
                            Cx3Map                            y)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index iy = 0; iy < FW; iy++) {
          Index const iiy = iy + c[1] - FW / 2;
          for (Index ix = 0; ix < FW; ix++) {
            Index const iix = ix + c[0] - FW / 2;
            y(ic, sample, trace) += sg(iix, iiy, ic, ib) * k(ix, iy) * b;
          }
        }
      }
    }
  }
};

template <int FW> struct GFunc<3, FW>
{
  using KT = FixedTensor<float, 3, FW>;

  inline static void
  Scatter(Eigen::Array<int16_t, 3, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx3CMap y, Cx5Map sg)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index iz = 0; iz < FW; iz++) {
        Index const iiz = iz + c[2] - FW / 2;
        for (Index iy = 0; iy < FW; iy++) {
          Index const iiy = iy + c[1] - FW / 2;
          for (Index ix = 0; ix < FW; ix++) {
            Index const iix = ix + c[0] - FW / 2;
            sg(iix, iiy, iiz, ic, 0) += y(ic, sample, trace) * k(ix, iy, iz);
          }
        }
      }
    }
  }

  inline static void
  Gather(Eigen::Array<int16_t, 3, 1> const c, int16_t const sample, int32_t const trace, KT const &k, Cx5CMap sg, Cx3Map y)
  {
    for (Index ic = 0; ic < y.dimension(0); ic++) {
      for (Index iz = 0; iz < FW; iz++) {
        Index const iiz = iz + c[2] - FW / 2;
        for (Index iy = 0; iy < FW; iy++) {
          Index const iiy = iy + c[1] - FW / 2;
          for (Index ix = 0; ix < FW; ix++) {
            Index const iix = ix + c[0] - FW / 2;
            y(ic, sample, trace) += sg(iix, iiy, iiz, ic, 0) * k(ix, iy, iz);
          }
        }
      }
    }
  }

  inline static void Scatter(Basis::CPtr                       basis,
                             Eigen::Array<int16_t, 3, 1> const c,
                             int16_t const                     sample,
                             int32_t const                     trace,
                             KT const                         &k,
                             Cx3CMap                           y,
                             Cx5Map                            sg)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index iz = 0; iz < FW; iz++) {
          Index const iiz = iz + c[2] - FW / 2;
          for (Index iy = 0; iy < FW; iy++) {
            Index const iiy = iy + c[1] - FW / 2;
            for (Index ix = 0; ix < FW; ix++) {
              Index const iix = ix + c[0] - FW / 2;
              sg(iix, iiy, iiz, ic, ib) += y(ic, sample, trace) * k(ix, iy, iz) * b;
            }
          }
        }
      }
    }
  }

  inline static void Gather(Basis::CPtr                       basis,
                            Eigen::Array<int16_t, 3, 1> const c,
                            int16_t const                     sample,
                            int32_t const                     trace,
                            KT const                         &k,
                            Cx5CMap                           sg,
                            Cx3Map                            y)
  {
    for (Index ib = 0; ib < basis->nB(); ib++) {
      auto const b = basis->entry(ib, sample, trace);
      for (Index ic = 0; ic < y.dimension(0); ic++) {
        for (Index iz = 0; iz < FW; iz++) {
          Index const iiz = iz + c[2] - FW / 2;
          for (Index iy = 0; iy < FW; iy++) {
            Index const iiy = iy + c[1] - FW / 2;
            for (Index ix = 0; ix < FW; ix++) {
              Index const iix = ix + c[0] - FW / 2;
              y(ic, sample, trace) += sg(iix, iiy, iiz, ic, ib) * k(ix, iy, iz) * b;
            }
          }
        }
      }
    }
  }
};

} // namespace rl