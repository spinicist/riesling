#pragma once

#include <Eigen/Dense>

Eigen::ArrayXf KB(Eigen::ArrayXf const &x, float const &beta);
float KB_FT(float const &u, float const &beta);
