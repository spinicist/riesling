#pragma once
#include <string>

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 1.f;
};

} // namespace rl
