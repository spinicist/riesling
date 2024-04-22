#include "diffs.hpp"

#include "threads.hpp"

namespace rl {

ForwardDiff::ForwardDiff(Index d)
  : Functor<Cx4>()
  , dim{d}
{
}

BackwardDiff::BackwardDiff(Index d)
  : Functor<Cx4>()
  , dim{d}
{
}

CentralDiff::CentralDiff(Index d)
  : Functor<Cx4>()
  , dim{d}
{
}

void ForwardDiff::operator()(Input x, Output y) const
{
  assert(x.dimensions() == y.dimensions());
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4       fwd{0, 1, 1, 1};
  fwd[dim + 1] = 2;
  y.setZero();
  y.slice(st1, sz).device(Threads::GlobalDevice()) = x.slice(fwd, sz) - x.slice(st1, sz);
}

void BackwardDiff::operator()(Input x, Output y) const
{
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4       bck{0, 1, 1, 1};
  bck[dim + 1] = 0;
  y.setZero();
  y.slice(st1, sz).device(Threads::GlobalDevice()) = x.slice(st1, sz) - x.slice(bck, sz);
}

void CentralDiff::operator()(Input x, Output y) const
{
  Sz4 const sz{x.dimension(0), x.dimension(1) - 2, x.dimension(2) - 2, x.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4       fwd{0, 1, 1, 1};
  Sz4       bck{0, 1, 1, 1};
  fwd[dim + 1] = 2;
  bck[dim + 1] = 0;
  y.setZero();
  y.slice(st1, sz).device(Threads::GlobalDevice()) = (x.slice(fwd, sz) - x.slice(bck, sz)) / x.slice(st1, sz).constant(2.f);
}

} // namespace rl
