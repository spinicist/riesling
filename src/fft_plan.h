#pragma once

#include "fft_util.h"
#include "types.h"

namespace FFT {

/* Templated wrapped around the fftw_plan_many_dft interface
 * Takes two template arguments - TensorRank and FFTRank. The last FFTRank dimensions will
 * be transformed, and the first (TensorRank - FFTRank) will not be transformed.
 */
template <int TensorRank, int FFTRank>
struct Plan
{
  static_assert(FFTRank <= TensorRank);
  using Tensor = Eigen::Tensor<Cx, TensorRank>;
  using TensorDims = typename Tensor::Dimensions;

  /*! Uses the specified Tensor as a workspace during planning
   */
  Plan(Tensor &workspace, Log log, long const nThreads = Threads::GlobalThreadCount());

  /*! Will allocate a workspace during planning
   */
  Plan(TensorDims const &dims, Log log, long const nThreads = Threads::GlobalThreadCount());

  ~Plan();

  void forward(Tensor &x) const; //!< Image space to k-space
  void reverse(Tensor &x) const; //!< K-space to image space

  float scale() const; //!< Return the scaling for the unitary transform

private:
  void plan(Tensor &x, long const nThreads);
  void applyPhase(Tensor &x, float const scale, bool const forward) const;

  TensorDims dims_;
  std::array<Cx1, FFTRank> phase_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log log_;
  bool threaded_;
};

using ThreeD = Plan<3, 3>;
using ThreeDMulti = Plan<4, 3>;
using ThreeDBasis = Plan<5, 3>;
using TwoDMulti = Plan<3, 2>;

} // namespace FFT
