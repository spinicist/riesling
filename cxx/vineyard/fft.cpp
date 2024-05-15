#include "fft.hpp"

#include "log.hpp"
#include "parse_args.hpp"
#include "tensors.hpp"
#include "threads.hpp"

#include "ducc0/fft/fftnd_impl.h"

namespace rl {
namespace FFT {

namespace internal {
auto Phase1D(Index const sz) -> Cx1
{
  Index const  c = sz / 2;
  double const shift = (double)c / sz;
  Rd1          ii(sz);
  std::iota(ii.data(), ii.data() + ii.size(), 0.);
  auto const s = ((ii - ii.constant(c / 2.)) * ii.constant(shift));
  Cxd1 const ph = ((s - s.floor()) * s.constant(2. * M_PI)).cast<Cxd>();
  Cx1 const  factors = (ph * ph.constant(Cxd{0., 1.})).exp().cast<Cx>();
  return factors;
}
} // namespace internal

template <int NFFT>
auto PhaseShift(Sz<NFFT> const shape) -> CxN<NFFT>
{
  Eigen::Tensor<Cx, NFFT> x(shape);
  x.setConstant(1.f);
  for (Index ii = 0; ii < NFFT; ii++) {
    auto const ph = internal::Phase1D(shape[ii]);
    Sz<NFFT>   rsh, brd;
    rsh.fill(1);
    brd = shape;
    rsh[ii] = shape[ii];
    brd[ii] = 1;
    rl::Log::Debug("x {} ph {} rsh {} brd {}", x.dimensions(), ph.dimensions(), rsh, brd);
    x.device(Threads::GlobalDevice()) = x * ph.reshape(rsh).broadcast(brd);
  }
  return x;
}

template <int ND, int NFFT>
void Forward(Eigen::TensorMap<CxN<ND>> &x, Sz<NFFT> const fftDims, Index const threads)
{
  auto const shape = x.dimensions();
  /* DUCC is row-major, reverse dims */
  std::vector<size_t> duccShape(ND), duccDims(NFFT);
  std::copy(shape.rbegin(), shape.rend(), duccShape.begin());
  std::transform(fftDims.begin(), fftDims.end(), duccDims.begin(), [](Index const d) { return ND - 1 - d; });
  float const scale = 1.f / std::sqrt(std::transform_reduce(duccDims.begin(), duccDims.end(), 1.f, std::multiplies{},
                                                            [duccShape](size_t const ii) { return duccShape[ii]; }));
  rl::Log::Debug("DUCC forward FFT shape {} dims {} scale {}", duccShape, duccDims, scale);
  ducc0::c2c(ducc0::cfmav(x.data(), duccShape), ducc0::vfmav(x.data(), duccShape), duccDims, true, scale, threads);
}

template <int ND>
void Forward(Eigen::TensorMap<CxN<ND>> &x, Index const threads)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Forward(x, dims, threads);
}

template <int ND, int NFFT>
void Forward(CxN<ND> &x, Sz<NFFT> const fftDims, Index const threads)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map, fftDims, threads);
}

template <int ND>
void Forward(CxN<ND> &x, Index const threads)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map, threads);
}

template <int ND, int NFFT>
void Adjoint(Eigen::TensorMap<CxN<ND>> &x, Sz<NFFT> const fftDims, Index const threads)
{
  auto const shape = x.dimensions();
  /* DUCC is row-major, reverse dims */
  std::vector<size_t> duccShape(ND), duccDims(NFFT);
  std::copy(shape.rbegin(), shape.rend(), duccShape.begin());
  std::transform(fftDims.begin(), fftDims.end(), duccDims.begin(), [](Index const d) { return ND - 1 - d; });
  float const scale = 1.f / std::sqrt(std::transform_reduce(duccDims.begin(), duccDims.end(), 1.f, std::multiplies{},
                                                            [duccShape](size_t const ii) { return duccShape[ii]; }));
  rl::Log::Debug("DUCC adjoint FFT shape {} dims {} scale {}", duccShape, duccDims, scale);
  ducc0::c2c(ducc0::cfmav(x.data(), duccShape), ducc0::vfmav(x.data(), duccShape), duccDims, false, scale, threads);
}

template <int ND>
void Adjoint(Eigen::TensorMap<CxN<ND>> &x, Index const threads)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Adjoint(x, dims, threads);
}

template <int ND, int NFFT>
void Adjoint(CxN<ND> &x, Sz<NFFT> const fftDims, Index const threads)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map, fftDims, threads);
}

template <int ND>
void Adjoint(CxN<ND> &x, Index const threads)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map, threads);
}

template auto PhaseShift<1>(Sz1 const) -> Cx1;
template auto PhaseShift<2>(Sz2 const) -> Cx2;
template auto PhaseShift<3>(Sz3 const) -> Cx3;

template void Forward<1, 1>(Cx1 &, Sz1 const, Index const);
template void Forward<3, 1>(Cx3 &, Sz1 const, Index const);
template void Forward<4, 2>(Cx4 &, Sz2 const, Index const);
template void Forward<4, 3>(Cx4 &, Sz3 const, Index const);
template void Forward<5, 3>(Cx5 &, Sz3 const, Index const);
template void Forward<1>(Cx1 &, Index const);
template void Forward<3>(Cx3 &, Index const);

template void Adjoint<3, 1>(Cx3 &, Sz1 const, Index const);
template void Adjoint<3, 2>(Cx3 &, Sz2 const, Index const);
template void Adjoint<4, 2>(Cx4 &, Sz2 const, Index const);
template void Adjoint<4, 3>(Cx4 &, Sz3 const, Index const);
template void Adjoint<5, 3>(Cx5 &, Sz3 const, Index const);
template void Adjoint<1>(Eigen::TensorMap<Cx1> &, Index const);
template void Adjoint<1>(Cx1 &, Index const);
template void Adjoint<2>(Cx2 &, Index const);
template void Adjoint<3>(Cx3 &, Index const);

} // namespace FFT
} // namespace rl
