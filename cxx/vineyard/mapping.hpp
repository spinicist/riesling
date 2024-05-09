#pragma once

#include "bucket.hpp"
#include "info.hpp"
#include "trajectory.hpp"
#include "types.hpp"

namespace rl {

struct NoncartesianIndex
{
  int32_t trace;
  int16_t sample;
};

template <int NDims>
struct Mapping
{
  Mapping(
    Trajectory const &t, float const nomOSamp, Index const kW, Index const bucketSize = 32, Index const splitSize = 16384);

  float     osamp;
  Sz2       noncartDims;
  Sz<NDims> cartDims, nomDims;

  std::vector<std::array<int16_t, NDims>>    cart;
  std::vector<NoncartesianIndex>             noncart;
  std::vector<Eigen::Array<float, NDims, 1>> offset;
  std::vector<Bucket<NDims>>                 buckets;
  std::vector<int32_t>                       sortedIndices;
};

} // namespace rl
