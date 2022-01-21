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
  Eigen::ArrayXf echoWeights;
};

struct Trajectory
{
  Trajectory();
  Trajectory(Info const &info, R3 const &points);
  Trajectory(Info const &info, R3 const &points, I1 const &echoes);
  Info const &info() const;
  R3 const &points() const;
  I1 const &echoes() const;

  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;

  // Generate an appropriate mapping
  Mapping mapping(
    Index const kw, float const os, float const inRes = -1.f, bool const shrink = false) const;

private:
  void init();

  Info info_;
  R3 points_;
  I1 echoes_;

  Eigen::ArrayXf mergeHi_, mergeLo_;
};
