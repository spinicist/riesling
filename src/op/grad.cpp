#include "grad.hpp"
#include "threads.hpp"
namespace rl {

GradOp::GradOp(InputDims const dims)
  : Parent("GradOp", dims, AddBack(dims, 3))
{
}

namespace {

template<typename T1, typename T2>
inline void ForwardDiff(Eigen::Index const d, T1 const &a, T2 &b)
{
  Sz4 sz{a.dimension(0), a.dimension(1), a.dimension(2), a.dimension(3)};
  Sz4 st{0, 0, 0, 0};
  Sz4 fwd{0, 0, 0, 0};
  fwd[d] = 1;
  sz[d] -= 1;
  b.chip(d - 1, 4).slice(st, sz).device(Threads::GlobalDevice()) = a.slice(fwd, sz) - a.slice(st, sz);
}

template<typename T1, typename T2>
inline void BackwardDiff(Eigen::Index const d, T1 const &a, T2 &b)
{
  Sz4 sz{a.dimension(0), a.dimension(1), a.dimension(2), a.dimension(3)};
  Sz4 st{0, 0, 0, 0};
  Sz4 bck{0, 0, 0, 0};
  st[d] = 1;
  sz[d] -= 1;

  b.slice(st, sz).device(Threads::GlobalDevice()) += a.chip(d - 1, 4).slice(bck, sz) - a.chip(d - 1, 4).slice(st, sz);
}
} // namespace

auto GradOp::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  y_.setZero();
  ForwardDiff(1, x, y_);
  ForwardDiff(2, x, y_);
  ForwardDiff(3, x, y_);

  this->finishForward(this->output(), time);
  return this->output();
}

auto GradOp::adjoint(OutputMap y) const -> InputMap
{
  auto const time = this->startAdjoint(y);
  x_.setZero();
  BackwardDiff(1, y, x_);
  BackwardDiff(2, y, x_);
  BackwardDiff(3, y, x_);
  this->finishAdjoint(this->input(), time);
  return this->input();
}

} // namespace rl
