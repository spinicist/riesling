#pragma once

#include "types.h"

namespace Sim {

struct Parameter
{
  float mean, std, lo, hi;
};

struct Tissue
{
  Tissue(std::vector<Parameter> const &pars);
  Eigen::ArrayXXf values(Index const NV);

private:
  Eigen::ArrayXf means, stds, los, his;
};

} // namespace Sim
