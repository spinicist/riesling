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
  Info::Type type;
  std::vector<CartesianIndex> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> echo;
  std::vector<float> sdc;
  std::vector<Point3> offset;
  std::vector<int32_t> sortedIndices;
  Sz3 cartDims, noncartDims;
  float osamp;
  int8_t echoes;
};

struct Trajectory
{
  Trajectory(Info const &info, R3 const &points, Log const &log);
  Trajectory(Info const &info, R3 const &points, I1 const &echoes, Log const &log);
  Info const &info() const;

  R3 const &points() const;
  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;

  // Return the appropriate filter value for merging lo/hi-res k-spaces
  float merge(int16_t const read, int32_t const spoke) const;

  // Generate an appropriate mapping
  Mapping mapping(
    float const os, Index const kRad, float const inRes = -1.f, bool const shrink = false) const;

private:
  void init();

  Info info_;
  R3 points_;
  I1 echoes_;
  Log log_;
  Eigen::ArrayXf mergeHi_, mergeLo_;
};
