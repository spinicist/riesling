#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "settings.hpp"
#include "types.hpp"

namespace rl {

struct T2Prep
{
  Settings seq;

  Index length() const;
  Eigen::ArrayXXf parameters(Index const nsamp) const;
  Eigen::ArrayXf simulate(Eigen::ArrayXf const &p) const;
};

struct T2InvPrep
{
  Settings seq;

  Index length() const;
  Eigen::ArrayXXf parameters(Index const nsamp) const;
  Eigen::ArrayXf simulate(Eigen::ArrayXf const &p) const;
};

} // namespace rl
