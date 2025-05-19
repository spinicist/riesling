#pragma once

#include <array>

namespace gw {

struct Info
{
  std::array<float, 3>  voxel_size, origin;
  std::array<float, 9> direction;
  float tr;
};


} // namespace rl
