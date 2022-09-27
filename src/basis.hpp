#pragma once

#include "io/writer.hpp"
#include "parse_args.hpp"
#include "types.hpp"

namespace rl {

struct Basis
{
  Basis(
    Eigen::ArrayXXf const &parameters,
    Eigen::ArrayXXf const &dynamics,
    float const thresh,
    Index const nB,
    bool const varimax);
  void write(HD5::Writer &writer);

  Eigen::ArrayXXf parameters, dynamics;
  Eigen::MatrixXf basis, dict;
  Eigen::ArrayXf norm;
};

} // namespace rl
