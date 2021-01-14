#pragma once

#include "fft_util.h"

/* 3D Fourier transform
 *
 */
struct FFT3
{
  FFT3(Cx3 &grid, Log &log, long const nThreads = Threads::GlobalThreadCount());
  ~FFT3();

  void forward() const; //!< Multiple images to multiple K-spaces
  void forward(Cx3 &x) const;
  void reverse() const; //!< Multiple K-spaces to multiple images
  void reverse(Cx3 &x) const;

private:
  void forwardPhase(Cx3 &x, float const scale) const;
  void reversePhase(Cx3 &x, float const scale) const;
  Cx3 &grid_;
  Cxd1 phX_, phY_, phZ_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
  bool threaded_;
};
