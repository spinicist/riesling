#pragma once

#include "types.hpp"

namespace rl {

struct Info
{
  Sz3             matrix;
  Eigen::Array3f  voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();
  float           tr = 1.f;
};

} // namespace rl
