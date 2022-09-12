#pragma once

#include "info.hpp"
#include "log.h"
#include "types.h"

namespace rl {
Cx3 SphericalPhantom(
  Sz3 const &matrix,
  Eigen::Array3f const &voxel_size,
  Eigen::Vector3f const &center,
  float const radius,
  float const intensity);
}
