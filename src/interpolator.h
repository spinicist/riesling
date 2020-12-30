#pragma once

#include "types.h"
#include <functional>
#include <vector>

struct InterpPair
{
  Size3 point;
  float weight;
};

struct Interpolator
{
  virtual long size() const = 0;
  virtual std::vector<InterpPair> weights(Point3 const offset) const = 0;
  virtual void apodize(Cx3 &img) const = 0;
  virtual void deapodize(Cx3 &img) const = 0;
};

struct NearestNeighbour final : Interpolator
{
  long size() const;
  std::vector<InterpPair> weights(Point3 const offset) const;
  void apodize(Cx3 &img) const;
  void deapodize(Cx3 &img) const;
};

struct KaiserBessel final : Interpolator
{
  long size() const;
  KaiserBessel(long const w, float const os, Dims3 const dims, bool const threeD);
  std::vector<InterpPair> weights(Point3 const offset) const;
  void apodize(Cx3 &img) const;
  void deapodize(Cx3 &img) const;

private:
  long w_, sz_;
  float beta_;
  R1 apodX_, apodY_, apodZ_;
  std::vector<Size3> points_;
};
