#include "grad.hpp"
#include "threads.hpp"
namespace rl {

GradOp::GradOp(InputDims const dims)
  : Parent("GradOp", dims, AddBack(dims, 3))
{
}

namespace {
inline auto ForwardDiff(Cx4 const &a, Eigen::Index const d)
{
  Sz4 const sz{a.dimension(0), a.dimension(1) - 2, a.dimension(2) - 2, a.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4 fwd{0, 1, 1, 1};
  fwd[d + 1] = 2;

  return (a.slice(fwd, sz) - a.slice(st1, sz));
}

inline auto BackwardDiff(Cx4 const &a, Eigen::Index const d)
{
  Sz4 const sz{a.dimension(0), a.dimension(1) - 2, a.dimension(2) - 2, a.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4 bck{0, 1, 1, 1};
  bck[d + 1] = 0;

  return (a.slice(st1, sz) - a.slice(bck, sz));
}
} // namespace

auto GradOp::forward(InputMap x) const -> OutputMap
{
  auto const time = this->startForward(x);
  this->output().setZero();
  auto dev = Threads::GlobalDevice();
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  this->output().chip<4>(0).slice(st1, sz).device(dev) = ForwardDiff(x, 0);
  this->output().chip<4>(1).slice(st1, sz).device(dev) = ForwardDiff(x, 1);
  this->output().chip<4>(2).slice(st1, sz).device(dev) = ForwardDiff(x, 2);

  this->finishForward(this->output(), time);
  return this->output();
}

auto GradOp::adjoint(OutputMap y) const -> InputMap
{
  auto const time = this->startAdjoint(y);
  this->input().setZero();
  auto dev = Threads::GlobalDevice();
  Sz4 const sz{y.dimension(0), y.dimension(1) - 2, y.dimension(2) - 2, y.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  this->input().slice(st1, sz).device(dev) =
    BackwardDiff(y.chip<4>(0), 0) + BackwardDiff(y.chip<4>(1), 1) + BackwardDiff(y.chip<4>(2), 2);
  this->finishAdjoint(this->input(), time);
  return this->input();
}

} // namespace rl
