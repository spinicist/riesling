#pragma once

#include "info.hpp"
#include "trajectory.hpp"
#include "types.hpp"

namespace rl {

template <int ND> struct Mapping
{
  template <typename T> using Array = Eigen::Array<T, ND, 1>;
  Array<int16_t> cart;
  int16_t        sample;
  int32_t        trace;
  Array<float>   offset;
  Array<int16_t> subgrid;
};

template <int ND> struct CalcMapping_t
{
  std::vector<Mapping<ND>> mappings;
  Sz2                      noncartDims;
  Sz<ND>                   cartDims;
};

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &t, float const nomOSamp, Index const kW, Index const subgridSize) -> CalcMapping_t<ND>;

inline auto SubgridFullwidth(Index const sgSize, Index const kW) { return sgSize + 2 * (kW / 2); }

template <int ND> inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd, Index const sgSz, Index const kW)
{
  Sz<ND> c;
  for (int id = 0; id < ND; id++) {
    c[id] = (sgSz * sgInd[id]) - (kW / 2);
  }
  return c;
}

} // namespace rl
