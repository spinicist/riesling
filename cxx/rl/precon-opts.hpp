#pragma once
#include <string>

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 0.f;
};

} // namespace rl
