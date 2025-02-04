#include "apodize.hpp"
#include "fft.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "tensors.hpp"
namespace rl {

template <int ND> auto Apodize(Sz<ND> const shape, typename TOps::Grid<ND>::Ptr const g) -> CxN<ND>
{
  CxN<ND>      k = g->kernel().template cast<Cx>();
  float const scale = std::sqrt(static_cast<float>(Product(shape)));
  Log::Debug("Apodiz", "Shape {} Grid shape {} Scale {}", shape, g->ishape, scale);
  k = k * k.constant(scale);
  CxN<ND> temp = TOps::Pad<Cx, ND>(k.dimensions(), FirstN<ND>(g->ishape)).forward(k);
  FFT::Adjoint(temp);
  ReN<ND> a = TOps::Pad<Cx, ND>(shape, temp.dimensions()).adjoint(temp).abs().real().cwiseMax(1.e-3f).inverse();
  // if constexpr (N == 3) { Log::Tensor("apodiz", a.dimensions(), a.data(), HD5::DimensionNames<3>{"i", "j", "k"}); }
  return a.template cast<Cx>();
}

template auto Apodize<1>(Sz1 const shape, typename TOps::Grid<1>::Ptr const g) -> Cx1;
template auto Apodize<2>(Sz2 const shape, typename TOps::Grid<2>::Ptr const g) -> Cx2;
template auto Apodize<3>(Sz3 const shape, typename TOps::Grid<3>::Ptr const g) -> Cx3;

} // namespace rl
