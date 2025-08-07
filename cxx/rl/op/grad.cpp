#include "grad.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {

/*
 * These difference functions have circular boundary conditions and are scaled to have max
 * eigenvalue = 1
 */

template <bool fwd, typename T1, typename T2, int N>
inline void ForwardDiff(T1 const &a, T2 &&b, Sz<N> const shape, Index const dim, float const s = 1.f)
{
  Sz<N> sz = shape, sz1 = shape, f0, fp1, fm1;
  fp1[dim] = 1;
  fm1[dim] = sz[dim] - 1;
  sz[dim] -= 1;
  sz1[dim] = 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward forward differences dim {}", dim, dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(fm1, sz1).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz1) - a.slice(fm1, sz1)) * b.slice(fm1, sz1).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint forward differences dim {}", dim, dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(f0, sz1).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz1) - a.slice(f0, sz1)) * b.slice(f0, sz1).constant(s / 2.f);
  }
}

template <bool fwd, typename T1, typename T2, int N>
inline void BackwardDiff(T1 const &a, T2 &&b, Sz<N> const shape, Index const dim, float const s = 1.f)
{
  Sz<N> sz = shape, sz1 = shape, f0, fp1, fm1;
  fp1[dim] = 1;
  fm1[dim] = sz[dim] - 1;
  sz[dim] -= 1;
  sz1[dim] = 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward backward differences dim {}", dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(f0, sz1).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz1) - a.slice(fm1, sz1)) * b.slice(fm1, sz1).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint backward differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
    b.slice(fm1, sz1).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz1) - a.slice(f0, sz1)) * b.slice(f0, sz1).constant(s / 2.f);
  }
}
} // namespace

template <int ND, int NG> Grad<ND, NG>::Grad(InDims const ish, Sz<NG> const d)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
{
}

template <int ND, int NG> auto Grad<ND, NG>::Make(InDims const ish, Sz<NG> const d) -> std::shared_ptr<Grad>
{
  return std::make_shared<Grad>(ish, d);
}

template <int ND, int NG> void Grad<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void Grad<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void Grad<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const  time = this->startForward(x, y, true);
  float const scale = s / std::sqrt(NG);
  for (Index ii = 0; ii < NG; ii++) {
    ForwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale);
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Grad<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(NG);
  for (Index ii = 0; ii < NG; ii++) {
    ForwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale);
  }
  this->finishAdjoint(x, time, true);
}

template struct Grad<1, 1>;
template struct Grad<2, 2>;
template struct Grad<3, 3>;

template struct Grad<5, 1>;
template struct Grad<5, 3>;

template <int ND, int NG> Div<ND, NG>::Div(OutDims const sh, Sz<NG> const d)
  : Parent("Grad", AddBack(sh, (Index)d.size()), sh)
  , dims_{d}
{
}

template <int ND, int NG> auto Div<ND, NG>::Make(OutDims const sh, Sz<NG> const d) -> std::shared_ptr<Div>
{
  return std::make_shared<Div>(sh, d);
}

template <int ND, int NG> void Div<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void Div<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void Div<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);

  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale);
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Div<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale);
  }
  this->finishAdjoint(x, time, true);
}

template struct Div<1, 1>;
template struct Div<2, 2>;
template struct Div<3, 3>;
template struct Div<5, 3>;

template <int ND, int NG> GradVec<ND, NG>::GradVec(InDims const ish, Sz<NG> const dims)
  : Parent("GradV", ish, AddBack(FirstN<ND - 1>(ish), (Index)((dims.size() * (dims.size() + 1)) / 2)))
  , dims_{dims}
{
  if ((Index)dims.size() != ishape[ND - 1]) {
    throw(Log::Failure("gradv", "Symmetrized gradient only, dims were {} and {}", dims.size(), ishape[ND - 1]));
  }
}

template <int ND, int NG> auto GradVec<ND, NG>::Make(InDims const ish, Sz<NG> const d) -> std::shared_ptr<GradVec>
{
  return std::make_shared<GradVec>(ish, d);
}

template <int ND, int NG> void GradVec<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void GradVec<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  Index       yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale);
    for (Index ij = ii + 1; ij < NG; ij++) {
      BackwardDiff<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
      BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
      yind++;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND, int NG> void GradVec<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void GradVec<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  Index       yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale);
    for (Index ij = ii + 1; ij < NG; ij++) {
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<2, 1>;
template struct GradVec<3, 2>;
template struct GradVec<4, 3>;
template struct GradVec<6, 3>;

} // namespace rl::TOps
