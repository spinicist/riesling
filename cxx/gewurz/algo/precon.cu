#include "precon.cuh"

#include "rl/log/log.hpp"
#include "rl/io/writer.hpp"

#include "../op/dft.cuh"
#include "dot.cuh"

#include <thrust/extrema.h>

namespace gw {

auto Preconditioner(DTensor<TDev, 3> const &T, int const nI, int const nJ, int const nK) -> DTensor<TDev, 2>
{
  rl::Log::Print("gewurz", "Preconditioner");
  auto const       nS = T.span.extent(1);
  auto const       nT = T.span.extent(2);
  DTensor<TDev, 2> M(nS, nT);
  thrust::fill(M.vec.begin(), M.vec.end(), TDev(1));

  DTensor<CuCx<TDev>, 2> Mks(nS, nT);
  DTensor<CuCx<TDev>, 3> Mimg(nI, nJ, nK);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), CuCx<TDev>(1));

  rl::HD5::Writer debug("debug.h5");

  gw::DFT::ThreeD dft{T.span};
  rl::Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  dft.adjoint(Mks.span, Mimg.span);
  rl::Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  dft.forward(Mimg.span, Mks.span);
  rl::Log::Print("Precon", "|img| {} |ks| {}", gw::CuNorm(Mimg.vec), gw::CuNorm(Mks.vec));
  float const λ = 0.0f;
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [λ] __device__(CuCx<TDev> x) { return (x == 0.f) ? 1.f : TDev(1 + λ) / (cuda::std::abs(x) + λ); });
  auto const mm = thrust::minmax_element(thrust::cuda::par, M.vec.begin(), M.vec.end());
  TDev const min = *(mm.first);
  TDev const max = *(mm.second);
  rl::Log::Print("Precon", "|M| {} Min {} Max {}", gw::CuNorm(M.vec), min, max);
  return M;
}

}