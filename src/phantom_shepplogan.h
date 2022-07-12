#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <math.h>

namespace rl {

Cx3 SheppLoganPhantom(
  Eigen::Array3l const &matrix,
  Eigen::Array3f const &voxel_size,
  Eigen::Vector3f const &center,
  Eigen::Vector3f const &rotation,
  float const radius,
  float const intensity);

}
