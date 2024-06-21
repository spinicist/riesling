#pragma once

#include "types.hpp"

#include "io/writer.hpp"

namespace rl {

template <typename Scalar> struct SVDBasis
{
  SVDBasis(
    Eigen::Array<Scalar, -1, -1> const &dynamics, Index const nB, bool const demean, bool const rotate, bool const normalize);
  Eigen::Matrix<Scalar, -1, -1> basis;
};

} // namespace rl