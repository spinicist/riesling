#pragma once

#include "Eigen/Core"
#include <unsupported/Eigen/Splines>

namespace rl {

/*
 * Eigen spline objects aren't very friendly. Wrap them in a class to do the required
 * scaling and transposes to get them working.
 */
struct Interpolator
{
  using Spline = Eigen::Spline<float, 1>;

  Interpolator();
  Interpolator(Eigen::ArrayXi const &x, Eigen::ArrayXf const &y, Eigen::Index const order = 3, bool const clamp = false);
  auto operator()(float const x) const -> float;
  auto operator()(Eigen::ArrayXi const &x) const -> Eigen::ArrayXf;

private:
  Spline m_spline;
  float  m_min, m_width;
  bool   m_clamp;
  auto   scale(float const x) const -> float;
  auto   scale(Eigen::ArrayXi const &x) const -> Eigen::ArrayXf;
};

} // namespace rl
