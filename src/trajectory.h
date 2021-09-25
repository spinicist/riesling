#pragma once

#include "info.h"
#include "log.h"
#include "types.h"

struct CartesianIndex
{
  int16_t x, y, z;
};

struct NoncartesianIndex
{
  int32_t spoke;
  int16_t read;
};

struct Mapping
{
  std::vector<CartesianIndex> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<float> sdc;
  std::vector<Point3> offset;
  std::vector<int32_t> sortedIndices;
  Sz3 cartDims;
  float osamp;
};

struct Trajectory
{
  Trajectory(Info const &info, R3 const &points, Log const &log);
  Info const &info() const;

  R3 const &points() const;
  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;

  // Return the appropriate filter value for merging lo/hi-res k-spaces
  float merge(int16_t const read, int32_t const spoke) const;

  // Generate an appropriate mapping
  Mapping mapping(
      float const os, long const kRad, float const inRes = -1.f, bool const shrink = false) const;

private:
  Info info_;
  R3 points_;
  Log log_;
  Eigen::ArrayXf mergeHi_, mergeLo_;
};
