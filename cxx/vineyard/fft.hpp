#pragma once

#include "types.hpp"

namespace rl {

namespace FFT {
/*
 * Phase ramps for FFT shifting
 */
template <int NFFT> auto PhaseShift(Sz<NFFT> const shape) -> CxN<NFFT>;

template <int ND, int NFFT> void Forward(Eigen::TensorMap<CxN<ND>> &data, Sz<NFFT> const fftDims, CxN<NFFT> const &ph);
template <int ND, int NFFT> void Forward(CxN<ND> &data, Sz<NFFT> const fftDims, CxN<NFFT> const &ph);
template <int ND> void           Forward(Eigen::TensorMap<CxN<ND>> &data, CxN<ND> const &ph);
template <int ND> void           Forward(CxN<ND> &data, CxN<ND> const &ph);

template <int ND, int NFFT> void Adjoint(Eigen::TensorMap<CxN<ND>> &data, Sz<NFFT> const fftDims, CxN<NFFT> const &ph);
template <int ND, int NFFT> void Adjoint(CxN<ND> &data, Sz<NFFT> const fftDims, CxN<NFFT> const &ph);
template <int ND> void           Adjoint(Eigen::TensorMap<CxN<ND>> &data, CxN<ND> const &ph);
template <int ND> void           Adjoint(CxN<ND> &data, CxN<ND> const &ph);

} // namespace FFT

} // namespace rl
