#pragma once

#include "info.h"
#include "log.h"
#include "types.h"

struct Trajectory
{
  Trajectory(Info const &info, R3 const &points, Log const &log);
  Info const &info() const;

  R3 const &points() const;
  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;
  float merge(int16_t const read, int32_t const spoke) const;

  Trajectory trim(float const res, Cx3 &data) const;

private:
  Info info_;
  R3 points_;
  Log log_;
  Eigen::ArrayXf mergeHi_, mergeLo_;
};
