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
  Index nP() const;
  Eigen::ArrayXXf values(Index const NV) const;

private:
  Eigen::ArrayXf means_, stds_, los_, his_;
};

struct Tissues
{
  Tissues(std::vector<Tissue> const &tissues);
  Index nP() const;
  Eigen::ArrayXXf values(Index const NV) const;

private:
  std::vector<Tissue> tissues_;
  Index nP_;
};

} // namespace Sim
