#pragma once

#include "types.h"

struct RadialInfo
{
  Array3l matrix;
  Eigen::Array3f voxel_size;
  long read_points;
  long read_gap;
  long spokes_hi;
  long spokes_lo;
  float lo_scale;
  long channels;
  long volumes = 1;
  float tr = 1.f;
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();

  inline long spokes_total() const
  {
    return spokes_hi + spokes_lo;
  }

  inline Cx2 radialChannel() const
  {
    Cx2 temp(read_points, spokes_total());
    temp.setZero();
    return temp;
  }

  inline Cx3 radialVolume() const
  {
    Cx3 temp(channels, read_points, spokes_total());
    temp.setZero();
    return temp;
  }

  inline Cx4 radialSeries() const
  {
    Cx4 temp(channels, read_points, spokes_total(), volumes);
    temp.setZero();
    return temp;
  }
};