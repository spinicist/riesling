#pragma once

#include "types.hpp"

#include "io/writer.hpp"

namespace rl {

void SaveSVDBasis(
  Eigen::ArrayXXf const &dynamics,
  float const            thresh,
  Index const            nB,
  bool const             demean,
  bool const             rotate,
  bool const             normalize,
  HD5::Writer           &writer);

}