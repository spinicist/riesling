#pragma once

#include "types.h"

Eigen::ArrayXf KB(Eigen::ArrayXf const &x, float const &beta);
float KB_FT(float const &u, float const &beta);
R2 KBKernel(Point2 const &offset, long const w, float const beta);
R3 KBKernel(Point3 const &offset, long const w, float const beta);
