#pragma once

#include "types.h"

struct Info
{
  enum struct Type : long
  {
    ThreeD = 1,
    ThreeDStack = 2
  };

  Array3l matrix;
  long read_points;
  long read_gap;
  long spokes_hi;
  long spokes_lo;
  float lo_scale;
  long channels;
  Type type = Type::ThreeD;
  long volumes = 1;
  long echoes = 1;
  float tr = 1.f;
  Eigen::Array3f voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();

  inline long spokes_total() const
  {
    return spokes_hi + spokes_lo;
  }

  inline float spoke_oversamp() const
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