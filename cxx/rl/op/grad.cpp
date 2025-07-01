#include "grad.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <bool fwd, typename T1, typename T2, typename SzT>
inline auto ForwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) f0, fp1;
  fp1[dim] = 1;
  sz[dim] -= 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward forward differences dim {}", dim, dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s);
  } else {
    Log::Debug("Grad", "Adjoint forward differences dim {}", dim, dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto BackwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) f0, fp1;
  fp1[dim] = 1;
  sz[dim] -= 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward backward differences dim {}", dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s);
  } else {
    Log::Debug("Grad", "Adjoint backward differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff0(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) fm1, f0, fp1;
  fp1[dim] = 2;
  fm1[dim] = 0;
  f0[dim] = 1;
  sz[dim] -= 2;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward central differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp1, sz) - a.slice(fm1, sz)) * b.slice(f0, sz).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint central differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s / 2.f);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff1(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{ // Thanks http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
  auto         sz = dims;
  decltype(sz) f0, fp1, fp2, fm1, fm2;
  f0[dim] = 2;
  fp1[dim] = 3;
  fp2[dim] = 4;
  fm1[dim] = 1;
  fm2[dim] = 0;
  sz[dim] -= 4;
  if constexpr (fwd) {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fm2, sz) - 8.f * a.slice(fm1, sz) + 8.f * a.slice(fp1, sz) - a.slice(fp2, sz)) *
      b.slice(f0, sz).constant(s / 12.f);
  } else {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp2, sz) - 8.f * a.slice(fp1, sz) + 8.f * a.slice(fm1, sz) - a.slice(fm2, sz)) *
      b.slice(f0, sz).constant(s / 12.f);
  }
}
} // namespace

template <int ND> Grad<ND>::Grad(InDims const ish, std::vector<Index> const &d)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
{
}

template <int ND> auto Grad<ND>::Make(InDims const ish, std::vector<Index> const &d) -> std::shared_ptr<Grad>
{
  return std::make_shared<Grad>(ish, d);
}

template <int ND> void Grad<ND>::forward(InCMap x, OutMap y) const
{
  y.setZero();
  iforward(x, y);
}

template <int ND> void Grad<ND>::adjoint(OutCMap y, InMap x) const
{
  x.setZero();
  iadjoint(y, x);
}

template <int ND> void Grad<ND>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    ForwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], s);
  }
  this->finishForward(y, time, true);
}

template <int ND> void Grad<ND>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    ForwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], s);
  }
  this->finishAdjoint(x, time, true);
}

template struct Grad<5>;

template <int ND> GradVec<ND>::GradVec(InDims const ishape, std::vector<Index> const &dims)
  : Parent("GradV", ishape, AddBack(FirstN<ND - 1>(ishape), (Index)((dims.size() * (dims.size() + 1)) / 2)))
  , dims_{dims}
{
  if ((Index)dims.size() != ishape[ND - 1]) {
    throw(Log::Failure("gradv", "Symmetrized gradient only, dims were {} and {}", dims.size(), ishape[ND - 1]));
  }
}

template <int ND> auto GradVec<ND>::Make(InDims const ish, std::vector<Index> const &d) -> std::shared_ptr<GradVec>
{
  return std::make_shared<GradVec>(ish, d);
}

template <int ND> void GradVec<ND>::forward(InCMap x, OutMap y) const
{
  y.setZero();
  iforward(x, y);
}

template <int ND> void GradVec<ND>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], s);
    for (Index ij = ii + 1; ij < (Index)dims_.size(); ij++) {
      BackwardDiff<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], s / 2.f);
      BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], s / 2.f);
    }
    yind++;
  }
  this->finishForward(y, time, false);
}

template <int ND> void GradVec<ND>::adjoint(OutCMap y, InMap x) const
{
  x.setZero();
  iadjoint(y, x);
}

template <int ND> void GradVec<ND>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    BackwardDiff<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], s);
    for (Index ij = ii + 1; ij < (Index)dims_.size(); ij++) {
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], s); /* No factor of 1/2 because the symmetrized matrix would have two elements */
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], s);
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<6>;

} // namespace rl::TOps
