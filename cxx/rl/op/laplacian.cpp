#include "laplacian.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <typename T1, typename T2, typename SzT> inline auto Laplace0(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{
  auto         sz = dims;
  decltype(sz) fm1, f0, fp1;
  fp1[dim] = 2;
  fm1[dim] = 0;
  f0[dim] = 1;
  sz[dim] -= 2;
  Log::Debug("Grad", "Laplacian dim {}", dim);
  b.slice(f0, sz).device(Threads::TensorDevice()) +=
    (a.slice(f0, sz) * a.slice(f0, sz).constant(1.f/3.f) - a.slice(fp1, sz) * a.slice(fp1, sz).constant(1.f/6.f) -
     a.slice(fm1, sz) * a.slice(fm1, sz).constant(1.f/6.f));
}
} // namespace

template <int ND> Laplacian<ND>::Laplacian(InDims const ish)
  : Parent("Laplacian", ish, ish)
{
}

template <int ND> auto Laplacian<ND>::Make(InDims const ish) -> std::shared_ptr<Laplacian>
{
  return std::make_shared<Laplacian>(ish);
}

template <int ND> void Laplacian<ND>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  CxN<ND>    temp(ishape);
  temp.setZero();
  Laplace0(x, temp, x.dimensions(), 0);
  Laplace0(x, temp, x.dimensions(), 1);
  Laplace0(x, temp, x.dimensions(), 2);
  y.device(Threads::TensorDevice()) = temp * temp.constant(-s);
  this->finishForward(y, time, false);
}

template <int ND> void Laplacian<ND>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  CxN<ND>    temp(ishape);
  temp.setZero();
  Laplace0(y, temp, y.dimensions(), 0);
  Laplace0(y, temp, y.dimensions(), 1);
  Laplace0(y, temp, y.dimensions(), 2);
  x.device(Threads::TensorDevice()) = temp * temp.constant(-s);
  this->finishAdjoint(x, time, false);
}

template <int ND> void Laplacian<ND>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  CxN<ND>    temp(ishape);
  temp.setZero();
  Laplace0(x, temp, x.dimensions(), 0);
  Laplace0(x, temp, x.dimensions(), 1);
  Laplace0(x, temp, x.dimensions(), 2);
  y.device(Threads::TensorDevice()) -= temp * temp.constant(s);
  this->finishForward(y, time, false);
}

template <int ND> void Laplacian<ND>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  CxN<ND>    temp(ishape);
  temp.setZero();
  Laplace0(y, temp, y.dimensions(), 0);
  Laplace0(y, temp, y.dimensions(), 1);
  Laplace0(y, temp, y.dimensions(), 2);
  x.device(Threads::TensorDevice()) -= temp * temp.constant(s);
  this->finishAdjoint(x, time, false);
}

template struct Laplacian<5>;

} // namespace rl::TOps
