#include "apodize.hpp"
#include "fft.hpp"
#include "kernel/kernel.hpp"
#include "log/log.hpp"
#include "op/pad.hpp"
#include "tensors.hpp"

namespace rl {

template <int ND, typename KF> auto Apodize(Sz<ND> const shape, Sz<ND> const gridshape, float const osamp) -> CxN<ND>
{
  Kernel<ND, KF> kernel(osamp);
  CxN<ND>        k = kernel().template cast<Cx>();
  float const    scale = std::sqrt(static_cast<float>(Product(shape)));
  Log::Debug("Apodiz", "Shape {} Grid shape {} Scale {}", shape, gridshape, scale);
  k = k * k.constant(scale);
  CxN<ND> temp = TOps::Pad<Cx, ND>(k.dimensions(), gridshape).forward(k);
  FFT::Adjoint(temp);
  ReN<ND> a = TOps::Pad<Cx, ND>(shape, temp.dimensions()).adjoint(temp).abs().real().cwiseMax(1.e-3f).inverse();
  // if constexpr (ND == 3) { Log::Tensor("apodiz", a.dimensions(), a.data(), HD5::DimensionNames<3>{"i", "j", "k"}); }
  return a.template cast<Cx>();
}

template auto Apodize<1, rl::ExpSemi<4>>(Sz1 const, Sz1 const, float const) -> Cx1;
template auto Apodize<2, rl::ExpSemi<4>>(Sz2 const, Sz2 const, float const) -> Cx2;
template auto Apodize<3, rl::ExpSemi<4>>(Sz3 const, Sz3 const, float const) -> Cx3;

} // namespace rl
