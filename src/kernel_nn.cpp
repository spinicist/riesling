#include "kernel_nn.h"

NearestNeighbour::NearestNeighbour(long const w)
    : w_{w}
{
}

long NearestNeighbour::radius() const
{
  return std::floor(w_ / 2.f);
}

Sz3 NearestNeighbour::start() const
{
  return Sz3{-(w_ - 1) / 2, -(w_ - 1) / 2, -(w_ - 1) / 2};
}

Sz3 NearestNeighbour::size() const
{
  return Sz3{w_, w_, w_};
}

R3 NearestNeighbour::kspace(Point3 const &offset) const
{
  R3 w(w_, w_, w_);
  w.setZero();
  w(w_ / 2, w_ / 2, w_ / 2) = 1.f;
  return w;
}

Cx3 NearestNeighbour::image(Point3 const &offset, Sz3 const &) const
{
  Cx3 w(w_, w_, w_);
  w.setConstant(1.f / sqrt(w_ * w_ * w_)); // Parseval's theorem
  return w;
}
