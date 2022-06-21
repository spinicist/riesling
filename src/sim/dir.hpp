#pragma once

#include "log.h"
#include "parameter.hpp"
#include "settings.hpp"
#include "types.h"

namespace rl {

struct DIR
{
  Settings seq;

  Index length() const;
  Eigen::ArrayXXf parameters(Index const nsamp) const;
  Eigen::ArrayXf simulate(Eigen::ArrayXf const &p) const;
};

} // namespace rl
