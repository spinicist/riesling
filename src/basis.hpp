#pragma once

#include "io/writer.hpp"
#include "parse_args.hpp"
#include "types.hpp"

namespace rl {

auto IdBasis() -> Re2;

void SaveBasis(
  Eigen::ArrayXXf const &dynamics,
  float const            thresh,
  Index const            nB,
  bool const             demean,
  bool const             rotate,
  bool const             normalize,
  HD5::Writer           &writer);

} // namespace rl
