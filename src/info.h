#pragma once

#include "types.h"

namespace rl {

struct Info
{
  Eigen::DSizes<Index, 3> matrix;
  Index channels;
  Index samples;
  Index traces;
  Index slabs = 1;
  bool grid3D = true;
  bool fft3D = true;
  Index frames = 1;
  Index volumes = 1;
  Eigen::Array3f voxel_size = Eigen::Vector3f::Ones();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Matrix3f direction = Eigen::Matrix3f::Identity();
  float tr = 1.f;

  inline Cx3 noncartesianVolume() const
  {
    Cx3 temp(channels, samples, traces);
    temp.setZero();
    return temp;
  }

  inline Cx4 noncartesianSeries() const
  {
    Cx4 temp(channels, samples, traces, volumes);
    temp.setZero();
    return temp;
  }

  inline Re3 trajectory() const
  {
    return Re3(3, samples, traces);
  }
};

} // namespace rl
