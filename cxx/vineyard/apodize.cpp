#include "apodize.hpp"
#include "fft.hpp"
#include "pad.hpp"

#include <cmath>

namespace rl {

template <int N>
auto Apodize(Sz<N> const shape, Sz<N> const gshape, std::shared_ptr<Kernel<Cx, N>> const &kernel) -> CxN<N>
{
  CxN<N> k = kernel->at(Eigen::Matrix<float, N, 1>::Zero()).template cast<Cx>();
  float const           scale = std::sqrt(static_cast<float>(Product(shape)));
  k = k * k.constant(scale);
  auto temp = Pad(k, gshape);
  FFT::Adjoint(temp);
  return Crop(temp, shape).inverse();
}

template auto Apodize<1>(Sz1 const shape, Sz1 const gshape, std::shared_ptr<Kernel<Cx, 1>> const &k) -> Cx1;
template auto Apodize<2>(Sz2 const shape, Sz2 const gshape, std::shared_ptr<Kernel<Cx, 2>> const &k) -> Cx2;
template auto Apodize<3>(Sz3 const shape, Sz3 const gshape, std::shared_ptr<Kernel<Cx, 3>> const &k) -> Cx3;

} // namespace rl::TOps
