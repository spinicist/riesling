#pragma once

#include "types.hpp"

#include "io/writer.hpp"

namespace rl {

struct SVDBasis
{
  SVDBasis(
    Eigen::Array<Cx, -1, -1> const &dynamics, Index const nB, bool const demean, bool const rotate, bool const normalize);
  Eigen::Matrix<Cx, -1, -1> basis;
};

} // namespace rl