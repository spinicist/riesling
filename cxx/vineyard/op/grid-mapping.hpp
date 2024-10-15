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
};

template <int ND>
inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd, Index const sgSz, Index const kW)
  -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * sgSz) - (kW / 2);
}

template <int ND> struct SubgridMapping
{
  template <typename T> using Array = Eigen::Array<T, ND, 1>;
  Array<int16_t>           corner;
  std::vector<Mapping<ND>> mappings;
};

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &t, Sz<ND> const &mat, Sz<ND> const &omat, Index const kW, Index const subgridSize)
  -> std::vector<SubgridMapping<ND>>;

inline auto SubgridFullwidth(Index const sgSize, Index const kW) { return sgSize + 2 * (kW / 2); }

} // namespace rl
