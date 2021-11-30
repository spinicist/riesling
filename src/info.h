#pragma once

#include "types.h"

struct Info
{
  enum struct Type : Index
  {
    ThreeD = 1,
    ThreeDStack = 2
  };

  Type type = Type::ThreeD;
  Index channels;
  Eigen::Array3l matrix;
  Index read_points;
  Index read_gap;
  Index spokes_hi;
  Index spokes_lo;
  float lo_scale;
  Index volumes = 1;
  Index echoes = 1;
  float tr = 1.f;
  Eigen::Array3f voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();

  inline Index spokes_total() const
  {
    return spokes_hi + spokes_lo;
  }

  inline float read_oversamp() const
  {
    return read_points / (matrix.maxCoeff() / 2);
  }

  inline Cx3 noncartesianVolume() const
  {
    Cx3 temp(channels, read_points, spokes_total());
    temp.setZero();
    return temp;
  }

  inline Cx4 noncartesianSeries() const
  {
    Cx4 temp(channels, read_points, spokes_total(), volumes);
    temp.setZero();
    return temp;
  }

  inline R3 trajectory() const
  {
    return R3(3, read_points, spokes_total());
  }
};