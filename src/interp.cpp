#include "interp.hpp"
#include "log.hpp"

#include <set>

namespace rl {

Interpolator::Interpolator()
  : m_min(0.)
  , m_width(0.)
{
}

// Sort input in ascending order

Interpolator::Interpolator(Eigen::ArrayXi const &x, Eigen::ArrayXf const &y, Eigen::Index const order, bool const c)
{
  if (x.size() != y.size()) { Log::Fail("Input vectors to spline must be same size"); }
  if (x.size() == 0) { Log::Fail("Cannot create a spline with no control points"); }
  if (order < 1) { Log::Fail("Interpolation order must be at least 1, was {}", order); }
  m_min = x[0];
  m_width = x[x.size() - 1] - m_min;
  m_clamp = c;
  Eigen::ArrayXf const scaledx = (x.cast<float>() - m_min) / m_width;
  m_spline = Eigen::SplineFitting<Spline>::Interpolate(y.transpose(), std::min<int>(x.rows() - 1, order), scaledx.transpose());
}

auto Interpolator::scale(float const x) const -> float
{
  auto y = (x - m_min) / m_width;
  if (m_clamp) {
    return std::clamp(y, 0.f, 1.f);
  } else {
    return y;
  }
}

auto Interpolator::scale(Eigen::ArrayXi const &x) const -> Eigen::ArrayXf
{
  auto y = (x.cast<float>() - m_min) / m_width;
  if (m_clamp) {
    return y.cwiseMax(0.f).cwiseMin(1.f);
  } else {
    return y;
  }
}

auto Interpolator::operator()(float const x) const -> float
{
  float const sx = scale(x);
  float const val = m_spline(sx)[0];
  return val;
}

auto Interpolator::operator()(const Eigen::ArrayXi &x) const -> Eigen::ArrayXf
{
  auto const     sx = scale(x);
  Eigen::ArrayXf output(sx.rows());
  for (Eigen::Index i = 0; i < sx.rows(); i++) {
    output[i] = m_spline(sx[i])[0];
  }
  return output;
}

} // namespace rl