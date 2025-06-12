#pragma once

#include "types.hpp"

namespace rl {
namespace FFT {

template <int ND, int NFFT> void Forward(Eigen::TensorMap<CxN<ND>> data, Sz<NFFT> const fftDims);
template <int ND> void           Forward(Eigen::TensorMap<CxN<ND>> &data);
template <int ND, int NFFT> void Forward(CxN<ND> &data, Sz<NFFT> const fftDims);
template <int ND> void           Forward(CxN<ND> &data);

template <int ND, int NFFT> void Adjoint(Eigen::TensorMap<CxN<ND>> data, Sz<NFFT> const fftDims);
template <int ND> void           Adjoint(Eigen::TensorMap<CxN<ND>> &data);
template <int ND, int NFFT> void Adjoint(CxN<ND> &data, Sz<NFFT> const fftDims);
template <int ND> void           Adjoint(CxN<ND> &data);

} // namespace FFT
} // namespace rl
