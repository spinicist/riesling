#pragma once

#include "types.hpp"

#include "io/writer.hpp"

namespace rl {

struct SVDBasis
{
  SVDBasis(Eigen::ArrayXXf const &dynamics,
           float const            thresh,
           Index const            nB,
           bool const             demean,
           bool const             rotate,
           bool const             normalize);
  Eigen::MatrixXf basis;
};

} // namespace rl