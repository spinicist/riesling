#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include "kernel.h"

struct Trajectory
{
  Trajectory();
  Trajectory(Info const &info, R3 const &points);
  Trajectory(Info const &info, R3 const &points, I1 const &frames);
  Info const &info() const;
  R3 const &points() const;
  I1 const &frames() const;
  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;
  std::tuple<Trajectory, Index> downsample(float const res, Index const lores, bool const shrink) const;

private:
  void init();

  Info info_;
  R3 points_;
  I1 frames_;
};
