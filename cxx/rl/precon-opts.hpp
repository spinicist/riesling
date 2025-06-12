#pragma once
#include <string>

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 1.e-3f;
};

} // namespace rl
