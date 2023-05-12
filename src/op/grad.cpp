#include "grad.hpp"
#include "threads.hpp"
namespace rl {

namespace {
template<typename T1, typename T2>
inline auto ForwardDiff(T1 const &a, T2 &&b, Sz4 const dims, Index const dim)
{
  Sz4 sz = dims;
  Sz4 st, fwd;
  fwd[dim] = 1;
  sz[dim] -= 1;
  b.slice(st, sz).device(Threads::GlobalDevice()) += (a.slice(fwd, sz) - a.slice(st, sz));
}

template<typename T1, typename T2>
inline auto BackwardDiff(T1 const &a, T2 &&b, Sz4 const dims, Index const dim)
{
  auto sz = dims;
  auto st = decltype(sz){};
  auto bck = decltype(sz){};
  st[dim] = 1;
  sz[dim] -= 1;
  b.slice(st, sz).device(Threads::GlobalDevice()) += (a.slice(bck, sz) - a.slice(st, sz));
}
} // namespace

GradOp::GradOp(InDims const ishape, std::vector<Index> const &d)
  : Parent("GradOp", ishape, AddBack(ishape, (Index)d.size())),
  dims_{d}
{
}

void GradOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    ForwardDiff(x, y.chip<4>(ii), x.dimensions(), dims_[ii]);
  }
  this->finishForward(y, time);
}

void GradOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    BackwardDiff(y.chip<4>(ii), x, x.dimensions(), dims_[ii]);
  }
  this->finishAdjoint(x, time);
}

GradVecOp::GradVecOp(InDims const dims)
  : Parent("GradVecOp", dims, AddBack(FirstN<4>(dims), 6))
{
}

void GradVecOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  Sz4 const sz = FirstN<4>(x.dimensions());
  y.setZero();
  for (Index ii = 0; ii < 3; ii++) {
    BackwardDiff(x.chip<4>(ii), y.chip<4>(ii), sz, ii + 1);
  }

  BackwardDiff(x.chip<4>(0), y.chip<4>(3), sz, 2);
  BackwardDiff(x.chip<4>(1), y.chip<4>(3), sz, 1);

  BackwardDiff(x.chip<4>(0), y.chip<4>(4), sz, 3);
  BackwardDiff(x.chip<4>(2), y.chip<4>(4), sz, 1);

  BackwardDiff(x.chip<4>(1), y.chip<4>(5), sz, 3);
  BackwardDiff(x.chip<4>(2), y.chip<4>(5), sz, 2);

  y.slice(Sz5{0, 0, 0, 0, 3}, AddBack(sz, 3)) /=  y.slice(Sz5{0, 0, 0, 0, 3}, AddBack(sz, 3)).constant(2.f);

  this->finishForward(y, time);
}

void GradVecOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  Sz4 const sz = FirstN<4>(x.dimensions());
  x.setZero();
  for (Index ii = 0; ii < 3; ii++) {
    ForwardDiff(y.chip<4>(ii), x.chip<4>(ii), sz, ii + 1);
  }
  ForwardDiff(y.chip<4>(3), x.chip<4>(0), sz, 2);
  ForwardDiff(y.chip<4>(4), x.chip<4>(0), sz, 3);

  ForwardDiff(y.chip<4>(3), x.chip<4>(1), sz, 1);
  ForwardDiff(y.chip<4>(5), x.chip<4>(1), sz, 3);

  ForwardDiff(y.chip<4>(4), x.chip<4>(2), sz, 1);
  ForwardDiff(y.chip<4>(5), x.chip<4>(2), sz, 2);

  this->finishAdjoint(x, time);
}

} // namespace rl
