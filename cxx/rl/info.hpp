#pragma once

#include "types.hpp"

namespace rl {

struct Info
{
  Eigen::Array3f  voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();
  float           tr = 1.f;
};

struct Transform
{
  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Array3f  δ = Eigen::Array3f::Zero();
};

} // namespace rl
