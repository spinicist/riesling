#include "grad.hpp"
#include "threads.hpp"
namespace rl {

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

GradOp::GradOp(InDims const dims)
  : Parent("GradOp", dims, AddBack(dims, 3))
{
}

void GradOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);

  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  auto dev = Threads::GlobalDevice();
  y.chip<4>(0).slice(st1, sz).device(dev) = ForwardDiff(x, 0);
  y.chip<4>(1).slice(st1, sz).device(dev) = ForwardDiff(x, 1);
  y.chip<4>(2).slice(st1, sz).device(dev) = ForwardDiff(x, 2);

  this->finishForward(y, time);
}

void GradOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  x.setZero();
  x.slice(st1, sz).device(Threads::GlobalDevice()) =
    BackwardDiff(y.chip<4>(0), 0) + BackwardDiff(y.chip<4>(1), 1) + BackwardDiff(y.chip<4>(2), 2);
  this->finishAdjoint(x, time);
}

Grad4Op::Grad4Op(InDims const dims)
  : Parent("Grad4Op", dims, AddBack(dims, 4))
{
}

void Grad4Op::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  // ForwardDiff(0, 0, x, y);
  // ForwardDiff(1, 1, x, y);
  // ForwardDiff(2, 2, x, y);
  // ForwardDiff(3, 3, x, y);
  this->finishForward(y, time);
}

void Grad4Op::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  // x.setZero();
  // BackwardDiff(0, 0, y, x);
  // BackwardDiff(1, 1, y, x);
  // BackwardDiff(2, 2, y, x);
  // BackwardDiff(3, 3, y, x);
  this->finishAdjoint(x, time);
}

GradVecOp::GradVecOp(InDims const dims)
  : Parent("GradVecOp", dims, AddBack(FirstN<4>(dims), 6))
{
}

void GradVecOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);

  auto dev = Threads::GlobalDevice();
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};

  y.chip<4>(0).slice(st1, sz).device(dev) = BackwardDiff(x.chip<4>(0), 0);
  y.chip<4>(1).slice(st1, sz).device(dev) = BackwardDiff(x.chip<4>(1), 1);
  y.chip<4>(2).slice(st1, sz).device(dev) = BackwardDiff(x.chip<4>(2), 2);

  y.chip<4>(3).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<4>(0), 1) + BackwardDiff(x.chip<4>(1), 0)) / y.chip<4>(3).slice(st1, sz).constant(2.f);

  y.chip<4>(4).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<4>(0), 2) + BackwardDiff(x.chip<4>(2), 0)) / y.chip<4>(4).slice(st1, sz).constant(2.f);

  y.chip<4>(5).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<4>(1), 2) + BackwardDiff(x.chip<4>(2), 1)) / y.chip<4>(5).slice(st1, sz).constant(2.f);

  this->finishForward(y, time);
}

void GradVecOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  auto dev = Threads::GlobalDevice();
  x.setZero();

  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  x.chip<4>(0).slice(st1, sz).device(dev) =
    ForwardDiff(y.chip<4>(0), 0) + ForwardDiff(y.chip<4>(3), 1) + ForwardDiff(y.chip<4>(4), 2);
  x.chip<4>(1).slice(st1, sz).device(dev) =
    ForwardDiff(y.chip<4>(3), 0) + ForwardDiff(y.chip<4>(1), 1) + ForwardDiff(y.chip<4>(5), 2);
  x.chip<4>(2).slice(st1, sz).device(dev) =
    ForwardDiff(y.chip<4>(4), 0) + ForwardDiff(y.chip<4>(5), 1) + ForwardDiff(y.chip<4>(2), 2);

  this->finishAdjoint(x, time);
}

} // namespace rl
