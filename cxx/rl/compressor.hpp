#pragma once

#include "log/log.hpp"
#include "types.hpp"

namespace rl {

struct Compressor
{
  Index out_channels() const;
  Cx4   compress(Cx4 const &source);
  Cx5   compress(Cx5 const &source);

  Eigen::MatrixXcf psi;
};
} // namespace rl
