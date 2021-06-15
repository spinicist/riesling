#pragma once

#include "fft_util.h"
#include "types.h"

namespace FFT {

/* Many ND Fourier transforms executed simultaneously
 * The "howmany" parameter for FFTW is the product of the tensor dimensions from 0 to FFTStartDim
 * The "stride" parameter for FFTW is the product of the tensor dimensions from
 * FFTStartDim + FFTNDim to the end
 */
template <int TensorRank, int FFTStart = 1, int FFTN = TensorRank - FFTStart>
struct Many
{
  static_assert(FFTStart + FFTN <= TensorRank);
  using Tensor = Eigen::Tensor<Cx, TensorRank>;
  using TensorDims = typename Tensor::Dimensions;

  Many(Tensor &workspace, Log &log, long const nThreads = Threads::GlobalThreadCount());
  ~Many();

  void forward() const;          //!< Image space to k-space
  void forward(Tensor &x) const; //!< Image space to k-space
  void reverse() const;          //!< K-space to image space
  void reverse(Tensor &x) const; //!< K-space to image space

  float scale() const; //!< Return the scaling for the unitary transform

private:
  void applyPhase(Tensor &x, float const scale, bool const forward) const;

  Tensor &ws_;
  std::array<Cx1, FFTN> phase_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
  bool threaded_;
};

} // namespace FFT
