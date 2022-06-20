#pragma once

#include "types.h"

namespace rl {

struct Parameter
{
  float mean, std, lo, hi;
  bool uniform = false;
};

struct Tissue
{
  Tissue(std::vector<Parameter> const pars);
  Index nP() const;
  Eigen::ArrayXXf values(Index const NV) const;

private:
  Eigen::ArrayXf means_, stds_, los_, his_;
  std::vector<bool> uni_;
};

struct Tissues
{
  Tissues(std::vector<Tissue> const tissues);
  Index nP() const;
  Eigen::ArrayXXf values(Index const NV) const;

private:
  std::vector<Tissue> tissues_;
  Index nP_;
};

extern Parameter const T1gm, T1wm, T1csf, T2gm, T2wm, T2csf, B1;

} // namespace rl
