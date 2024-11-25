#include "grad.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <typename T1, typename T2, typename SzT> inline auto ForwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{
  auto         sz = dims;
  decltype(sz) st, fwd;
  fwd[dim] = 1;
  sz[dim] -= 1;
  b.slice(st, sz).device(Threads::TensorDevice()) += (a.slice(fwd, sz) - a.slice(st, sz));
}

template <typename T1, typename T2, typename SzT> inline auto BackwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{
  auto sz = dims;
  auto st = decltype(sz){};
  auto bck = decltype(sz){};
  st[dim] = 1;
  sz[dim] -= 1;
  b.slice(st, sz).device(Threads::TensorDevice()) += (a.slice(bck, sz) - a.slice(st, sz));
}
} // namespace

template <int ND>
Grad<ND>::Grad(InDims const ish, std::vector<Index> const &d)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
{
}

template <int ND> void Grad<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    ForwardDiff(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii]);
  }
  this->finishForward(y, time, false);
}

template <int ND> void Grad<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    BackwardDiff(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii]);
  }
  this->finishAdjoint(x, time, false);
}

template struct Grad<5>;

template <int ND>
GradVec<ND>::GradVec(InDims const ishape, std::vector<Index> const &dims)
  : Parent("GradVec", ishape, AddBack(FirstN<ND - 1>(ishape), ND * (ND - 1) / 2))
  , dims_{dims}
{
}

template <int ND> void GradVec<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  y.setZero();
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < dims_.size(); ii++) {
    BackwardDiff(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii]);
    for (Index ij = ii + 1; ij < dims_.size(); ij++) {
      BackwardDiff(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii]);
      BackwardDiff(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij]);
      y.template chip<ND - 1>(yind) /= y.template chip<ND - 1>(yind).constant(2.f);
      yind++;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND> void GradVec<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  x.setZero();
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < dims_.size(); ii++) {
    ForwardDiff(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii]);
    for (Index ij = ii + 1; ij < dims_.size(); ij++) {
      ForwardDiff(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij]);
      ForwardDiff(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii]);
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<6>;

} // namespace rl::TOps
