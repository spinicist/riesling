#pragma once

#include "../types.h"

namespace rl {
namespace FFT {
/* Templated wrapped around the fftw_plan_many_dft interface
 * Takes two template arguments - TensorRank and FFTRank. The last FFTRank dimensions will
 * be transformed, and the first (TensorRank - FFTRank) will not be transformed.
 */
template <int TensorRank, int FFTRank>
struct FFT
{
  static_assert(FFTRank <= TensorRank);
  using Tensor = Eigen::Tensor<Cx, TensorRank>;
  using TensorDims = typename Tensor::Dimensions;

  virtual ~FFT() {}

  virtual void forward(Tensor &x) const = 0; //!< Image space to k-space
  virtual void reverse(Tensor &x) const = 0; //!< K-space to image space
};

void Start();
void End();
void SetTimelimit(double time);
Cx1 Phase(Index const sz);

template <int TRank, int FFTRank>
std::unique_ptr<FFT<TRank, FFTRank>>
Make(typename FFT<TRank, FFTRank>::TensorDims const &dims, Index const threads = 0);

template <int TRank, int FFTRank>
std::unique_ptr<FFT<TRank, FFTRank>>
Make(typename FFT<TRank, FFTRank>::Tensor &ws, Index const threads = 0);

} // namespace FFT
}
