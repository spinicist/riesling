#pragma once

#include "fft_util.h"

/* 1D Real-To-Complex Transform
 *
 */
struct FFT1DReal2Complex
{
  FFT1DReal2Complex(long const N, Log &log);
  ~FFT1DReal2Complex();

  Eigen::ArrayXcf forward(Eigen::ArrayXf const &in) const; //!< Real to complex
  Eigen::ArrayXf reverse(Eigen::ArrayXcf const &in) const; //!< Complex to real

private:
  long const N_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
};
