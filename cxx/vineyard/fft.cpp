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

/*
 *  Wrapper for DUCC. Adapted from https://github.com/tensorflow/tensorflow/blob/master/third_party/ducc/threading.h
 */
struct ThreadPool : ducc0::detail_threading::thread_pool
{
  ThreadPool(Eigen::ThreadPoolDevice &dev)
    : pool_{dev.getPool()}
  {
  }

  size_t nthreads() const override { return pool_->NumThreads(); }
  size_t adjust_nthreads(size_t nthreads_in) const override
  {
    // If called by a thread in the pool, return 1
    if (pool_->CurrentThreadId() >= 0) {
      return 1;
    } else if (nthreads_in == 0) {
      return pool_->NumThreads();
    }
    return std::min<size_t>(nthreads_in, pool_->NumThreads());
  };
  void submit(std::function<void()> work) override { pool_->Schedule(std::move(work)); }

private:
  Eigen::ThreadPoolInterface *pool_;
};
using Guard = ducc0::detail_threading::ScopedUseThreadPool;
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
    x.device(Threads::GlobalDevice()) = x * ph.reshape(rsh).broadcast(brd);
  }
  return x;
}

template <int ND, int NFFT>
void Forward(Eigen::TensorMap<CxN<ND>> &x, Sz<NFFT> const fftDims, CxN<NFFT> const &ph)
{
  auto const shape = x.dimensions();
  /* For phase ramps */
  Sz<ND> rsh, brd;
  rsh.fill(1);
  brd.fill(1);
  for (Index ii = 0; ii < ND; ii++) {
    if (std::find(fftDims.begin(), fftDims.end(), ii) == fftDims.end()) {
      brd[ii] = shape[ii];
    } else {
      rsh[ii] = shape[ii];
    }
  }

  /* DUCC is row-major, reverse dims */
  std::vector<size_t> duccShape(ND), duccDims(NFFT);
  std::copy(shape.rbegin(), shape.rend(), duccShape.begin());
  std::transform(fftDims.begin(), fftDims.end(), duccDims.begin(), [](Index const d) { return ND - 1 - d; });
  float const scale = 1.f / std::sqrt(std::transform_reduce(duccDims.begin(), duccDims.end(), 1.f, std::multiplies{},
                                                            [duccShape](size_t const ii) { return duccShape[ii]; }));
  internal::ThreadPool pool(Threads::GlobalDevice());
  internal::Guard      guard(pool);
  rl::Log::Debug("FFT Shift");
  x.device(Threads::GlobalDevice()) = x * ph.reshape(rsh).broadcast(brd);
  rl::Log::Debug("DUCC forward FFT shape {} dims {} scale {}", duccShape, duccDims, scale);
  ducc0::c2c(ducc0::cfmav(x.data(), duccShape), ducc0::vfmav(x.data(), duccShape), duccDims, true, scale, pool.nthreads());
  rl::Log::Debug("FFT Shift");
  x.device(Threads::GlobalDevice()) = x * ph.reshape(rsh).broadcast(brd);
}

template <int ND>
void Forward(Eigen::TensorMap<CxN<ND>> &x, CxN<ND> const &ph)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Forward(x, dims, ph);
}

template <int ND, int NFFT>
void Forward(CxN<ND> &x, Sz<NFFT> const fftDims, CxN<NFFT> const &ph)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map, fftDims, ph);
}

template <int ND>
void Forward(CxN<ND> &x, CxN<ND> const &ph)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map, ph);
}

template <int ND, int NFFT>
void Adjoint(Eigen::TensorMap<CxN<ND>> &x, Sz<NFFT> const fftDims, CxN<NFFT> const &ph)
{
  auto const shape = x.dimensions();
  /* For phase ramps */
  Sz<ND> rsh, brd;
  rsh.fill(1);
  brd.fill(1);
  for (Index ii = 0; ii < ND; ii++) {
    if (std::find(fftDims.begin(), fftDims.end(), ii) == fftDims.end()) {
      brd[ii] = shape[ii];
    } else {
      rsh[ii] = shape[ii];
    }
  }

  /* DUCC is row-major, reverse dims */
  std::vector<size_t> duccShape(ND), duccDims(NFFT);
  std::copy(shape.rbegin(), shape.rend(), duccShape.begin());
  std::transform(fftDims.begin(), fftDims.end(), duccDims.begin(), [](Index const d) { return ND - 1 - d; });
  float const scale = 1.f / std::sqrt(std::transform_reduce(duccDims.begin(), duccDims.end(), 1.f, std::multiplies{},
                                                            [duccShape](size_t const ii) { return duccShape[ii]; }));
  
  internal::ThreadPool pool(Threads::GlobalDevice());
  internal::Guard      guard(pool);
  rl::Log::Debug("FFT Shift");
  x.device(Threads::GlobalDevice()) = x / ph.reshape(rsh).broadcast(brd);
  rl::Log::Debug("DUCC adjoint FFT shape {} dims {} scale {}", duccShape, duccDims, scale);
  ducc0::c2c(ducc0::cfmav(x.data(), duccShape), ducc0::vfmav(x.data(), duccShape), duccDims, false, scale, pool.nthreads());
  rl::Log::Debug("FFT Shift");
  x.device(Threads::GlobalDevice()) = x / ph.reshape(rsh).broadcast(brd);
}

template <int ND>
void Adjoint(Eigen::TensorMap<CxN<ND>> &x, CxN<ND> const &ph)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Adjoint(x, dims, ph);
}

template <int ND, int NFFT>
void Adjoint(CxN<ND> &x, Sz<NFFT> const fftDims, CxN<NFFT> const &ph)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map, fftDims, ph);
}

template <int ND>
void Adjoint(CxN<ND> &x, CxN<ND> const &ph)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map, ph);
}

template auto PhaseShift<1>(Sz1 const) -> Cx1;
template auto PhaseShift<2>(Sz2 const) -> Cx2;
template auto PhaseShift<3>(Sz3 const) -> Cx3;

template void Forward<1, 1>(Cx1 &, Sz1 const, Cx1 const &ph);
template void Forward<3, 1>(Cx3 &, Sz1 const, Cx1 const &ph);
template void Forward<4, 1>(Cx4 &, Sz1 const, Cx1 const &ph);
template void Forward<4, 2>(Cx4 &, Sz2 const, Cx2 const &ph);
template void Forward<4, 3>(Cx4 &, Sz3 const, Cx3 const &ph);
template void Forward<5, 2>(Cx5 &, Sz2 const, Cx2 const &ph);
template void Forward<5, 3>(Cx5 &, Sz3 const, Cx3 const &ph);
template void Forward<6, 3>(Cx6 &, Sz3 const, Cx3 const &ph);
template void Forward<1>(Cx1 &, Cx1 const &ph);
template void Forward<3>(Cx3 &, Cx3 const &ph);

template void Adjoint<3, 1>(Cx3 &, Sz1 const, Cx1 const &ph);
template void Adjoint<3, 2>(Cx3 &, Sz2 const, Cx2 const &ph);
template void Adjoint<4, 1>(Cx4 &, Sz1 const, Cx1 const &ph);
template void Adjoint<4, 2>(Cx4 &, Sz2 const, Cx2 const &ph);
template void Adjoint<4, 3>(Cx4 &, Sz3 const, Cx3 const &ph);
template void Adjoint<5, 2>(Cx5 &, Sz2 const, Cx2 const &ph);
template void Adjoint<5, 3>(Cx5 &, Sz3 const, Cx3 const &ph);
template void Adjoint<6, 3>(Cx6 &, Sz3 const, Cx3 const &ph);
template void Adjoint<1>(Eigen::TensorMap<Cx1> &, Cx1 const &ph);
template void Adjoint<1>(Cx1 &, Cx1 const &ph);
template void Adjoint<2>(Cx2 &, Cx2 const &ph);
template void Adjoint<3>(Cx3 &, Cx3 const &ph);

} // namespace FFT
} // namespace rl
