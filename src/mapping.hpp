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

template <size_t Rank>
struct Mapping
{
  struct Bucket
  {
    Sz<Rank>             gridSize, minCorner, maxCorner;
    std::vector<int32_t> indices;

    auto empty() const -> bool;
    auto size() const -> Index;
    auto bucketSize() const -> Sz<Rank>;
    auto bucketStart() const -> Sz<Rank>;
    auto gridStart() const -> Sz<Rank>;
    auto sliceSize() const -> Sz<Rank>;
  };

  Mapping(
    Trajectory const &t,
    float const       nomOSamp,
    Index const       kW,
    Index const       bucketSize = 32,
    Index const       splitSize = 16384);

  float    osamp;
  Sz2      noncartDims;
  Sz<Rank> cartDims, nomDims;

  std::vector<std::array<int16_t, Rank>>    cart;
  std::vector<NoncartesianIndex>            noncart;
  std::vector<Eigen::Array<float, Rank, 1>> offset;
  std::vector<Bucket>                       buckets;
  std::vector<int32_t>                      sortedIndices;
};

} // namespace rl
