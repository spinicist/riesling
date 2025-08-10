#include "fft.hpp"

#include "log/log.hpp"
#include "sys/threads.hpp"
#include "tensors.hpp"

#include "ducc0/fft/fftnd_impl.h"

// #include "experimental/mdspan"

#include "fmt/std.h"

namespace rl {
namespace FFT {
namespace internal {
/*
 *  Wrapper for DUCC. Adapted from https://github.com/tensorflow/tensorflow/blob/master/third_party/ducc/threading.h
 */
struct ThreadPool : ducc0::detail_threading::thread_pool
{
  ThreadPool(Eigen::ThreadPoolDevice &dev)
    : pool_{dev.getPool()}
  {
  }

  size_t nthreads() const override { return (size_t)pool_->NumThreads(); }
  size_t adjust_nthreads(size_t nthreads_in) const override
  {
    // If called by a thread in the pool, return 1
    if (pool_->CurrentThreadId() >= 0) {
      return 1;
    } else if (nthreads_in == 0) {
      return (size_t)pool_->NumThreads();
    }
    return std::min<size_t>(nthreads_in, (size_t)pool_->NumThreads());
  };
  void submit(std::function<void()> work) override { pool_->Schedule(std::move(work)); }

private:
  Eigen::ThreadPoolInterface *pool_;
};
using Guard = ducc0::detail_threading::ScopedUseThreadPool;
} // namespace internal

void Shift(ducc0::vfmav<Cx> const &x, ducc0::fmav_info::shape_t const &axes)
{
  auto const ND = x.ndim();
  auto const N = 1 << (axes.size() - 1);

  auto task = [&](Index const nlo, Index const nhi) {
    for (Index in = nlo; in < nhi; in++) {
      std::vector<ducc0::slice> lslice(ND), rslice(ND);
      for (size_t ia = 0; ia < axes.size(); ia++) {
        auto const a = axes[ia];
        if (x.shape()[a] > 1) {
          if (x.shape()[a] % 2 != 0) { throw Log::Failure("FFT", "Shape {} dim {} was not even", x.shape(), a); }
          auto const mid = x.shape()[a] / 2;
          if ((in >> ia) & 1) {
            lslice[a].end = mid;
            rslice[a].beg = mid;
          } else {
            lslice[a].beg = mid;
            rslice[a].end = mid;
          }
        }
      }
      ducc0::mav_apply([](Cx &a, Cx &b) { std::swap(a, b); }, 1, x.subarray(lslice), x.subarray(rslice));
    }
  };
  Threads::ChunkFor(task, N);
}

template <int ND, int NFFT> void Run(Eigen::TensorMap<CxN<ND>> &x, Sz<NFFT> const fftDims, bool const fwd)
{
  auto const shape = x.dimensions();
  /* DUCC is row-major, reverse dims */
  std::vector<size_t> duccShape(ND), duccDims(NFFT);
  std::copy(shape.rbegin(), shape.rend(), duccShape.begin());
  std::transform(fftDims.begin(), fftDims.end(), duccDims.begin(), [](Index const d) { return ND - 1 - d; });
  float const scale = 1.f / std::sqrt(std::transform_reduce(duccDims.cbegin(), duccDims.cend(), 1.f, std::multiplies{},
                                                            [duccShape](size_t const ii) { return duccShape[ii]; }));
  rl::Log::Debug("FFT", "{} Shape {} dims {} scale {}", fwd ? "Forward" : "Adjoint", shape, fftDims, scale);
  internal::ThreadPool pool(Threads::TensorDevice());
  internal::Guard      guard(pool);
  ducc0::cfmav         xc(x.data(), duccShape);
  ducc0::vfmav         xv(x.data(), duccShape);
  auto                 t = Log::Now();
  Shift(xv, duccDims);
  rl::Log::Debug("FFT", "Shift took {}", Log::ToNow(t));
  t = Log::Now();
  ducc0::c2c(xc, xv, duccDims, fwd, scale, pool.nthreads());
  rl::Log::Debug("FFT", "{} took {}", fwd ? "Forward" : "Adjoint", Log::ToNow(t));
  t = Log::Now();
  Shift(xv, duccDims);
  rl::Log::Debug("FFT", "Shift took {}", Log::ToNow(t));
}

template <int ND, int NFFT> void Forward(Eigen::TensorMap<CxN<ND>> x, Sz<NFFT> const fftDims) { Run(x, fftDims, true); }

template <int ND, int NFFT> void Forward(CxN<ND> &x, Sz<NFFT> const fftDims)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map, fftDims);
}

template <int ND> void Forward(Eigen::TensorMap<CxN<ND>> &x)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Forward(x, dims);
}

template <int ND> void Forward(CxN<ND> &x)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Forward(map);
}

template <int ND, int NFFT> void Adjoint(Eigen::TensorMap<CxN<ND>> x, Sz<NFFT> const fftDims) { Run(x, fftDims, false); }

template <int ND, int NFFT> void Adjoint(CxN<ND> &x, Sz<NFFT> const fftDims)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map, fftDims);
}

template <int ND> void Adjoint(Eigen::TensorMap<CxN<ND>> &x)
{
  Sz<ND> dims;
  std::iota(dims.begin(), dims.end(), 0);
  Adjoint(x, dims);
}

template <int ND> void Adjoint(CxN<ND> &x)
{
  Eigen::TensorMap<CxN<ND>> map(x.data(), x.dimensions());
  Adjoint(map);
}

template void Forward<2, 1>(Cx2Map, Sz1 const);
template void Forward<3, 1>(Cx3Map, Sz1 const);
template void Forward<3, 2>(Cx3Map, Sz2 const);
template void Forward<4, 2>(Cx4Map, Sz2 const);
template void Forward<4, 3>(Cx4Map, Sz3 const);
template void Forward<5, 1>(Cx5Map, Sz1 const);
template void Forward<5, 2>(Cx5Map, Sz2 const);
template void Forward<5, 3>(Cx5Map, Sz3 const);
template void Forward<1, 1>(Cx1 &, Sz1 const);
template void Forward<2, 1>(Cx2 &, Sz1 const);
template void Forward<3, 1>(Cx3 &, Sz1 const);
template void Forward<3, 2>(Cx3 &, Sz2 const);
template void Forward<4, 1>(Cx4 &, Sz1 const);
template void Forward<4, 2>(Cx4 &, Sz2 const);
template void Forward<4, 3>(Cx4 &, Sz3 const);
template void Forward<5, 1>(Cx5 &, Sz1 const);
template void Forward<5, 2>(Cx5 &, Sz2 const);
template void Forward<5, 3>(Cx5 &, Sz3 const);
template void Forward<6, 3>(Cx6 &, Sz3 const);
template void Forward<1>(Cx1 &);
template void Forward<3>(Cx3 &);

template void Adjoint<2, 1>(Cx2Map, Sz1 const);
template void Adjoint<3, 1>(Cx3Map, Sz1 const);
template void Adjoint<3, 2>(Cx3Map, Sz2 const);
template void Adjoint<4, 2>(Cx4Map, Sz2 const);
template void Adjoint<4, 3>(Cx4Map, Sz3 const);
template void Adjoint<5, 1>(Cx5Map, Sz1 const);
template void Adjoint<5, 2>(Cx5Map, Sz2 const);
template void Adjoint<5, 3>(Cx5Map, Sz3 const);
template void Adjoint<2, 1>(Cx2 &, Sz1 const);
template void Adjoint<3, 1>(Cx3 &, Sz1 const);
template void Adjoint<3, 2>(Cx3 &, Sz2 const);
template void Adjoint<4, 1>(Cx4 &, Sz1 const);
template void Adjoint<4, 2>(Cx4 &, Sz2 const);
template void Adjoint<4, 3>(Cx4 &, Sz3 const);
template void Adjoint<5, 1>(Cx5 &, Sz1 const);
template void Adjoint<5, 2>(Cx5 &, Sz2 const);
template void Adjoint<5, 3>(Cx5 &, Sz3 const);
template void Adjoint<6, 3>(Cx6 &, Sz3 const);
template void Adjoint<1>(Eigen::TensorMap<Cx1> &);
template void Adjoint<1>(Cx1 &);
template void Adjoint<2>(Cx2 &);
template void Adjoint<3>(Cx3 &);

} // namespace FFT
} // namespace rl
