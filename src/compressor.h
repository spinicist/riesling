#pragma once

#include "log.h"
#include "types.h"

namespace rl {

struct Compressor
{
  Index out_channels() const;
  void compress(Cx4 const &source, Cx4 &dest);
  Cx3 compress(Cx3 const &source);
  Cx4 compress(Cx4 const &source);

  Eigen::MatrixXcf psi;
};
}
