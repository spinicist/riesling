#include "sense.h"
#include "../tensorOps.h"
#include "../threads.h"

SenseOp::SenseOp(Output &maps, Output::Dimensions const &bigSize)
    : maps_{std::move(maps)}
{
  auto const smallSize = maps_.dimensions();
  std::copy_n(bigSize.begin(), 4, full_.begin());
  std::copy_n(smallSize.begin(), 4, size_.begin());
  std::transform(
      bigSize.begin(), bigSize.end(), smallSize.begin(), left_.begin(), [](long big, long small) {
        return (big - small + 1) / 2;
      });
  std::transform(
      bigSize.begin(), bigSize.end(), smallSize.begin(), right_.begin(), [](long big, long small) {
        return (big - small) / 2;
      });
}

void SenseOp::A(Input const &x, Output &y) const
{
  assert(x.dimension(0) == maps_.dimension(1));
  assert(x.dimension(1) == maps_.dimension(2));
  assert(x.dimension(2) == maps_.dimension(3));
  assert(y.dimension(0) == maps_.dimension(0));
  assert(y.dimension(1) == maps_.dimension(1));
  assert(y.dimension(2) == maps_.dimension(2));
  assert(y.dimension(3) == maps_.dimension(3));

  Eigen::IndexList<Eigen::type2index<1>, int, int, int> res;
  res.set(1, x.dimension(0));
  res.set(2, x.dimension(1));
  res.set(3, x.dimension(2));
  Eigen::IndexList<int, Eigen::type2index<1>, Eigen::type2index<1>, Eigen::type2index<1>> brd;
  brd.set(0, maps_.dimension(0));

  Eigen::array<std::pair<int, int>, 4> paddings;
  std::transform(
      left_.begin(), left_.end(), right_.begin(), paddings.begin(), [](long left, long right) {
        return std::make_pair(left, right);
      });

  y.device(Threads::GlobalDevice()) = (x.reshape(res).broadcast(brd) * maps_).pad(paddings);
}

void SenseOp::Adj(Output const &x, Input &y) const
{
  assert(x.dimension(0) == maps_.dimension(0));
  assert(x.dimension(1) == maps_.dimension(1));
  assert(x.dimension(2) == maps_.dimension(2));
  assert(x.dimension(3) == maps_.dimension(3));
  assert(y.dimension(0) == maps_.dimension(1));
  assert(y.dimension(1) == maps_.dimension(2));
  assert(y.dimension(2) == maps_.dimension(3));
  y.device(Threads::GlobalDevice()) = ConjugateSum(x.slice(left_, size_), maps_);
}

void SenseOp::AdjA(Input const &x, Input &y) const
{
  y.device(Threads::GlobalDevice()) = x;
}
