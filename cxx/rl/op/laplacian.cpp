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
    (a.slice(f0, sz) * a.slice(f0, sz).constant(1.f / 3.f) - a.slice(fp1, sz) * a.slice(fp1, sz).constant(1.f / 6.f) -
     a.slice(fm1, sz) * a.slice(fm1, sz).constant(1.f / 6.f));
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
  y.device(Threads::TensorDevice()) = temp * temp.constant(-s/2.f);
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
  x.device(Threads::TensorDevice()) = temp * temp.constant(-s/2.f);
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
  y.device(Threads::TensorDevice()) -= temp * temp.constant(s/2.f);
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
  x.device(Threads::TensorDevice()) -= temp * temp.constant(s/2.f);
  this->finishAdjoint(x, time, false);
}

template struct Laplacian<3>;
template struct Laplacian<5>;

template <int ND> IsoΔ3D<ND>::IsoΔ3D(InDims const ish)
  : Parent("IsoΔ3D", ish, ish)
{
}

template <int ND> auto IsoΔ3D<ND>::Make(InDims const ish) -> std::shared_ptr<IsoΔ3D> { return std::make_shared<IsoΔ3D>(ish); }

template <int ND> void IsoΔ3D<ND>::doIt(InCMap x, OutMap y, float const s) const
{
  auto const  time = this->startForward(x, y, false);

  Sz<ND> sz = x.dimensions(), so;
  for (Index ii = 0; ii < 3; ii++) {
    sz[ii] -= 2;
    so[ii] = 1;
  }

  std::array<Sz<ND>, 6> NN;
  std::array<Sz<ND>, 12> NNN;
  std::array<Sz<ND>, 8> NNNN;

  std::fill(NN.begin(), NN.end(), so);
  std::fill(NNN.begin(), NNN.end(), so);
  std::fill(NNNN.begin(), NNNN.end(), so);

  Index inn = 0, innn = 0, innnn = 0;
  for (Index ii = 0; ii < 3; ii++) {
    NN[inn++][ii] = 0;
    NN[inn++][ii] = 2;
    for (Index ij = ii + 1; ij < 3; ij++) {
      NNN[innn][ii] = 0;
      NNN[innn++][ij] = 0;
      NNN[innn][ii] = 0;
      NNN[innn++][ij] = 2;
      NNN[innn][ii] = 2;
      NNN[innn++][ij] = 0;
      NNN[innn][ii] = 2;
      NNN[innn++][ij] = 2;
      for (Index ik = ij + 1; ik < 3; ik++) {
        NNNN[innnn][ii] = 0;
        NNNN[innnn][ij] = 0;
        NNNN[innnn++][ik] = 0;
        NNNN[innnn][ii] = 0;
        NNNN[innnn][ij] = 0;
        NNNN[innnn++][ik] = 2;
        NNNN[innnn][ii] = 0;
        NNNN[innnn][ij] = 2;
        NNNN[innnn++][ik] = 0;
        NNNN[innnn][ii] = 2;
        NNNN[innnn][ij] = 0;
        NNNN[innnn++][ik] = 0;
        NNNN[innnn][ii] = 0;
        NNNN[innnn][ij] = 2;
        NNNN[innnn++][ik] = 2;
        NNNN[innnn][ii] = 2;
        NNNN[innnn][ij] = 0;
        NNNN[innnn++][ik] = 2;
        NNNN[innnn][ii] = 2;
        NNNN[innnn][ij] = 2;
        NNNN[innnn++][ik] = 0;
        NNNN[innnn][ii] = 2;
        NNNN[innnn][ij] = 2;
        NNNN[innnn++][ik] = 2;
      }
    }
  }

  auto o = y.slice(so, sz);
  o.device(Threads::TensorDevice()) +=
    o.constant(s / (48.f * std::sqrt(28.3))) *
    (o.constant(-200.f) * x.slice(so, sz) +
     o.constant(20.f) * (x.slice(NN[0], sz) + x.slice(NN[1], sz) + x.slice(NN[2], sz) + x.slice(NN[3], sz) +
                      x.slice(NN[4], sz) + x.slice(NN[5], sz)) +
     o.constant(6.f) * (x.slice(NNN[0], sz) + x.slice(NNN[1], sz) + x.slice(NNN[2], sz) + x.slice(NNN[3], sz) +
                      x.slice(NNN[4], sz) + x.slice(NNN[5], sz) + x.slice(NNN[6], sz) + x.slice(NNN[7], sz) +
                      x.slice(NNN[8], sz) + x.slice(NNN[9], sz) + x.slice(NNN[10], sz) + x.slice(NNN[11], sz)) +
     o.constant(1.f) * (x.slice(NNNN[0], sz) + x.slice(NNNN[1], sz) + x.slice(NNNN[2], sz) + x.slice(NNNN[3], sz) +
                      x.slice(NNNN[4], sz) + x.slice(NNNN[5], sz) + x.slice(NNNN[6], sz) + x.slice(NNNN[7], sz)));

  this->finishForward(y, time, false);
}

template <int ND> void IsoΔ3D<ND>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.setZero();
  doIt(x, y, s);
  this->finishForward(y, time, false);
}

template <int ND> void IsoΔ3D<ND>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.setZero();
  doIt(y, x, s);
  this->finishAdjoint(x, time, false);
}

template <int ND> void IsoΔ3D<ND>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  doIt(x, y, s);
  this->finishForward(y, time, false);
}

template <int ND> void IsoΔ3D<ND>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  doIt(y, x, s);
  this->finishAdjoint(x, time, false);
}

template struct IsoΔ3D<3>;
template struct IsoΔ3D<5>;

} // namespace rl::TOps
