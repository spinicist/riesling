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
    Sz<Rank> minCorner, maxCorner;
    std::vector<int32_t> indices;

    auto empty() const -> bool;
    auto size() const -> Index;
    auto gridSize() const -> Sz<Rank>;
  };

  Mapping(
    Trajectory const &t,
    Index const kW,
    float const nomOSamp,
    Index const bucketSize = 32,
    Index const splitSize = 16384,
    Index const read0 = 0);

  float osamp;
  Sz2 noncartDims;
  Sz<Rank> cartDims;
  int8_t frames;

  std::vector<std::array<int16_t, Rank>> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> frame;
  std::vector<Eigen::Array<float, Rank, 1>> offset;
  std::vector<Bucket> buckets;
  std::vector<int32_t> sortedIndices;
};

} // namespace rl
