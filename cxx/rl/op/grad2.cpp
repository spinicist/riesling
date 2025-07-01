#include "grad2.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND> Grad2<ND>::Grad2(InDims const ish)
  : Parent("Grad2", ish, ish)
  , pad(ishape, Concatenate(MulToEven(FirstN<3>(ish), 2), LastN<ND - 3>(ish)))
{
}

template <int ND> auto Grad2<ND>::Make(InDims const ish) -> std::shared_ptr<Grad2> { return std::make_shared<Grad2>(ish); }

template <int ND> void Grad2<ND>::filter(OutMap x) const
{
  Index const N0 = x.dimension(0);
  Index const N1 = x.dimension(1);
  Index const N2 = x.dimension(2);
  for (Index i2 = 0; i2 < N2; i2++) {
    float const k2 = (i2 - N2 / 2.f);
    for (Index i1 = 0; i1 < N1; i1++) {
      float const k1 = (i1 - N1 / 2.f);
      for (Index i0 = 0; i0 < N0; i0++) {
        float const k0 = (i0 - N0 / 2.f);
        float const f = (k0 * k0 + k1 * k1 + k2 * k2);
        x.template chip<2>(i2).template chip<1>(i1).template chip<0>(i0) =
          x.template chip<2>(i2).template chip<1>(i1).template chip<0>(i0) * Cx(-f, 0);
      }
    }
  }
}

template <int ND> void Grad2<ND>::forward(InCMap x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  CxN<ND>    temp = pad.forward(x);
  rl::FFT::Forward(temp, Sz3{0, 1, 2});
  filter(temp);
  rl::FFT::Adjoint(temp, Sz3{0, 1, 2});
  y = pad.adjoint(temp);
  this->finishForward(y, time, false);
}

template <int ND> void Grad2<ND>::adjoint(OutCMap y, InMap x) const {}

template <int ND> void Grad2<ND>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InTensor   temp = x;
  rl::FFT::Forward(temp, Sz3{0, 1, 2});
  filter(temp);
  rl::FFT::Forward(temp, Sz3{0, 1, 2});
  y += temp * Cx(s);
  this->finishForward(y, time, true);
}

template <int ND> void Grad2<ND>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  this->finishAdjoint(x, time, true);
}

template struct Grad2<5>;

} // namespace rl::TOps
