#pragma once

#include "types.hpp"

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
  using TensorMap = Eigen::TensorMap<Tensor>;

  virtual ~FFT() {}

  virtual void forward(TensorMap x) const = 0; //!< Image space to k-space
  virtual void reverse(TensorMap x) const = 0; //!< K-space to image space
};

void Start(std::string const &execname);
void End(std::string const &execname);
void SetTimelimit(double time);
Cx1  Phase(Index const sz);

template <int TRank, int FFTRank>
std::shared_ptr<FFT<TRank, FFTRank>> Make(typename FFT<TRank, FFTRank>::TensorDims const &dims, Index const threads = 0);

template <int TRank, int FFTRank>
std::shared_ptr<FFT<TRank, FFTRank>> Make(typename FFT<TRank, FFTRank>::TensorMap ws, Index const threads = 0);

} // namespace FFT
} // namespace rl
