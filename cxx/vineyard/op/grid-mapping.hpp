#pragma once

#include "info.hpp"
#include "trajectory.hpp"
#include "types.hpp"

namespace rl {

struct NoncartesianIndex
{
  int32_t trace;
  int16_t sample;
};

template <int ND> struct Mapping
{
  Sz2    noncartDims;
  Sz<ND> cartDims;

  std::array<int16_t, ND>    cart;
  NoncartesianIndex          noncart;
  Eigen::Array<float, ND, 1> offset;
  Sz<ND>                     subgrid;
};

template <int ND> struct CalcMapping_t
{
  std::vector<Mapping<ND>> mappings;
  Sz2                      noncartDims;
  Sz<ND>                   cartDims;
};

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &t, float const nomOSamp, Index const kW, Index const subgridSize)
  -> CalcMapping_t<ND>;

} // namespace rl
