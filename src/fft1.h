#pragma once

#include "fft_util.h"

/* 1D Real-To-Complex Transform
 *
 */
struct FFT1DReal2Complex
{
  FFT1DReal2Complex(Index const N, Log &log);
  ~FFT1DReal2Complex();

  Cx1 forward(R1 const &in) const; //!< Real to complex
  R1 reverse(Cx1 const &in) const; //!< Complex to real
  void shift(R1 &in) const;
  void shift(Cx1 &in) const;

private:
  Index const N_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
};
