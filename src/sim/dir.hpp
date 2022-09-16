#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "settings.hpp"
#include "types.hpp"

namespace rl {

struct DIR
{
  Settings seq;
  DIR(Settings const s);

  Index length() const;
  Eigen::ArrayXXf parameters(Index const nsamp) const;
  Eigen::ArrayXf simulate(Eigen::ArrayXf const &p) const;
};

} // namespace rl
