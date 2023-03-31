#include "grad.hpp"
#include "threads.hpp"
namespace rl {

namespace {
template <typename T1, typename T2>
inline void ForwardDiff(Eigen::Index const dim, Eigen::Index const ind, T1 const &a, T2 &b)
{
  auto sz = a.dimensions();
  auto st = decltype(sz){};
  auto fwd = decltype(sz){};
  fwd[dim] = 1;
  sz[dim] -= 1;
  auto const d = T2::NumDimensions - 1;
  b.chip(ind, d).slice(st, sz).device(Threads::GlobalDevice()) = a.slice(fwd, sz) - a.slice(st, sz);
}

template <typename T1, typename T2>
inline void BackwardDiff(Eigen::Index const dim, Eigen::Index const ind, T1 const &a, T2 &b)
{
  auto sz = b.dimensions();
  auto st = decltype(sz){};
  auto bck = decltype(sz){};
  st[dim] = 1;
  sz[dim] -= 1;
  auto const d = T1::NumDimensions - 1;
  b.slice(st, sz).device(Threads::GlobalDevice()) += a.chip(ind, d).slice(bck, sz) - a.chip(ind, d).slice(st, sz);
}
} // namespace

GradOp::GradOp(InputDims const dims)
  : Parent("GradOp", dims, AddBack(dims, 3))
{
}

auto GradOp::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  y_.setZero();
  ForwardDiff(1, 0, x, y_);
  ForwardDiff(2, 1, x, y_);
  ForwardDiff(3, 2, x, y_);

  this->finishForward(this->output(), time);
  return this->output();
}

auto GradOp::adjoint(OutputMap y) const -> InputMap
{
  auto const time = this->startAdjoint(y);
  x_.setZero();
  BackwardDiff(1, 0, y, x_);
  BackwardDiff(2, 1, y, x_);
  BackwardDiff(3, 2, y, x_);
  this->finishAdjoint(this->input(), time);
  return this->input();
}

Grad4Op::Grad4Op(InputDims const dims)
  : Parent("Grad4Op", dims, AddBack(dims, 4))
{
}

auto Grad4Op::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  y_.setZero();
  ForwardDiff(0, 0, x, y_);
  ForwardDiff(1, 1, x, y_);
  ForwardDiff(2, 2, x, y_);
  ForwardDiff(3, 3, x, y_);

  this->finishForward(this->output(), time);
  return this->output();
}

auto Grad4Op::adjoint(OutputMap y) const -> InputMap
{
  auto const time = this->startAdjoint(y);
  x_.setZero();
  BackwardDiff(0, 0, y, x_);
  BackwardDiff(1, 1, y, x_);
  BackwardDiff(2, 2, y, x_);
  BackwardDiff(3, 3, y, x_);
  this->finishAdjoint(this->input(), time);
  return this->input();
}

} // namespace rl
