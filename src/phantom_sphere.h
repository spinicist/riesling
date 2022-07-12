#pragma once

#include "info.h"
#include "log.h"
#include "types.h"

namespace rl {
Cx3 SphericalPhantom(
  Eigen::Array3l const &matrix,
  Eigen::Array3f const &voxel_size,
  Eigen::Vector3f const &center,
  float const radius,
  float const intensity);
}
