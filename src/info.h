#pragma once

#include "types.h"

namespace rl {

struct Info
{
  enum struct Type : Index
  {
    ThreeD = 1,
    ThreeDStack = 2,
    TwoD = 3
  };

  Type type = Type::ThreeD;
  Eigen::Array3l matrix;
  Index channels;
  Index read_points;
  Index spokes;
  Index volumes = 1;
  Index frames = 1;
  float tr = 1.f;
  Eigen::Array3f voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();

  inline Cx3 noncartesianVolume() const
  {
    Cx3 temp(channels, read_points, spokes);
    temp.setZero();
    return temp;
  }

  inline Cx4 noncartesianSeries() const
  {
    Cx4 temp(channels, read_points, spokes, volumes);
    temp.setZero();
    return temp;
  }

  inline R3 trajectory() const
  {
    return R3(3, read_points, spokes);
  }
};

} // namespace rl
