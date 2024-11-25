#include "apodize.hpp"
#include "fft.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "tensors.hpp"
namespace rl {

template <int N> auto Apodize(Sz<N> const shape, Sz<N> const gshape, std::shared_ptr<KernelBase<Cx, N>> const &kernel) -> CxN<N>
{
  CxN<N>      k = kernel->operator()(KernelBase<Cx, N>::Point::Zero()).template cast<Cx>();
  float const scale = std::sqrt(static_cast<float>(Product(shape)));
  Log::Debug("Apodiz", "Shape {} Grid shape {} Scale {}", shape, gshape, scale);
  k = k * k.constant(scale);
  CxN<N> temp = TOps::Pad<Cx, N>(k.dimensions(), gshape).forward(k);
  FFT::Adjoint(temp);
  ReN<N> a = TOps::Pad<Cx, N>(shape, temp.dimensions()).adjoint(temp).abs().real().cwiseMax(1.e-3f).inverse();
  // if constexpr (N == 3) { Log::Tensor("apodiz", a.dimensions(), a.data(), HD5::DimensionNames<3>{"i", "j", "k"}); }
  return a.template cast<Cx>();
}

template auto Apodize<1>(Sz1 const shape, Sz1 const gshape, std::shared_ptr<KernelBase<Cx, 1>> const &k) -> Cx1;
template auto Apodize<2>(Sz2 const shape, Sz2 const gshape, std::shared_ptr<KernelBase<Cx, 2>> const &k) -> Cx2;
template auto Apodize<3>(Sz3 const shape, Sz3 const gshape, std::shared_ptr<KernelBase<Cx, 3>> const &k) -> Cx3;

} // namespace rl
