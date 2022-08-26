#pragma once

#include "info.h"
#include "trajectory.h"
#include "types.h"

namespace rl {

struct NoncartesianIndex
{
  int32_t trace;
  int16_t sample;
};

template <size_t NDims>
struct Mapping
{
  using Sz = Eigen::DSizes<Index, NDims>;
  struct Bucket
  {
    Sz minCorner, maxCorner;
    std::vector<int32_t> indices;

    auto empty() const -> bool;
    auto size() const -> Index;
    auto gridSize() const -> Sz;
  };

  Mapping(
    Trajectory const &t,
    Index const kW,
    float const osamp,
    Index const bucketSize = 32,
    Index const read0 = 0);

  bool fft3D;
  Sz2 noncartDims;
  Sz cartDims;
  int8_t frames;
  Eigen::ArrayXf frameWeights;

  std::vector<std::array<int16_t, NDims>> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> frame;
  std::vector<Eigen::Array<float, NDims, 1>> offset;
  std::vector<Bucket> buckets;
  std::vector<int32_t> sortedIndices;
};

} // namespace rl
