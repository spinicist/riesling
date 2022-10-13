#include "diffs.hpp"

namespace rl {

ForwardDiff::ForwardDiff(Index d)
  : Functor<Cx4>()
  , dim{d} {};
BackwardDiff::BackwardDiff(Index d)
  : Functor<Cx4>()
  , dim{d} {};
CentralDiff::CentralDiff(Index d)
  : Functor<Cx4>()
  , dim{d} {};
TotalDiff::TotalDiff()
  : Functor<Cx4>()
  , x_{0}
  , y_{1}
  , z_{2} {};

auto ForwardDiff::operator()(Cx4 const &a) const -> Cx4 const &
{
  Sz4 const sz{a.dimension(0), a.dimension(1) - 2, a.dimension(2) - 2, a.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4 fwd{0, 1, 1, 1};
  fwd[dim + 1] = 2;

  Cx4 b(a.dimensions());
  b.setZero();
  b.slice(st1, sz) = a.slice(fwd, sz) - a.slice(st1, sz);
  return b;
}

auto BackwardDiff::operator()(Cx4 const &a) const -> Cx4 const &
{
  Sz4 const sz{a.dimension(0), a.dimension(1) - 2, a.dimension(2) - 2, a.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4 bck{0, 1, 1, 1};
  bck[dim + 1] = 0;

  Cx4 b(a.dimensions());
  b.setZero();
  b.slice(st1, sz) = a.slice(st1, sz) - a.slice(bck, sz);
  return b;
}

auto CentralDiff::operator()(Cx4 const &a) const -> Cx4 const &
{
  Sz4 const sz{a.dimension(0), a.dimension(1) - 2, a.dimension(2) - 2, a.dimension(3) - 2};
  Sz4 const st1{0, 1, 1, 1};
  Sz4 fwd{0, 1, 1, 1};
  Sz4 bck{0, 1, 1, 1};
  fwd[dim + 1] = 2;
  bck[dim + 1] = 0;

  Cx4 b(a.dimensions());
  b.setZero();
  b.slice(st1, sz) = (a.slice(fwd, sz) - a.slice(bck, sz)) / a.slice(st1, sz).constant(2.f);
  return b;
}

auto TotalDiff::operator()(Cx4 const &a) const -> Cx4 const &
{
  Cx4 b = x_(a) + y_(a) + z_(a);
  return b;
}

} // namespace rl
