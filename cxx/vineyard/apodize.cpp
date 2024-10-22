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
  fmt::print(stderr, "k {} |k| {}\n", k.dimensions(), Norm(k));
  CxN<N> temp = TOps::Pad<Cx, N>(k.dimensions(), gshape).forward(k);
  fmt::print(stderr, "t {} |t| {}\n", temp.dimensions(), Norm(temp));
  FFT::Adjoint(temp);
  fmt::print(stderr, "f {} |f| {}\n", temp.dimensions(), Norm(temp));
  CxN<N> a = TOps::Crop<Cx, N>(temp.dimensions(), shape).forward(temp);
  fmt::print(stderr, "a {} |a| {}\n", a.dimensions(), Norm(a));
  a.device(Threads::TensorDevice()) = a.inverse();
  fmt::print(stderr, "i {} |i| {}\n", a.dimensions(), Norm(a));
  return a;
}

template auto Apodize<1>(Sz1 const shape, Sz1 const gshape, std::shared_ptr<KernelBase<Cx, 1>> const &k) -> Cx1;
template auto Apodize<2>(Sz2 const shape, Sz2 const gshape, std::shared_ptr<KernelBase<Cx, 2>> const &k) -> Cx2;
template auto Apodize<3>(Sz3 const shape, Sz3 const gshape, std::shared_ptr<KernelBase<Cx, 3>> const &k) -> Cx3;

} // namespace rl
