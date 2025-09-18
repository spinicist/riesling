#pragma once
#include <string>

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       max = 1.f;
};

} // namespace rl
