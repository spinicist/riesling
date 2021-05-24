#pragma once

#include "fft_util.h"

/* Multiple 3D Fourier transforms executed simultaneously
 *
 */
struct FFT3N
{
  FFT3N(Cx4 &grid, Log &log, long const nThreads = Threads::GlobalThreadCount());
  ~FFT3N();

  void forward() const;       //!< Multiple images to multiple k-spaces
  void forward(Cx4 &x) const; //!< Multiple images to multiple k-spaces
  void reverse() const;       //!< Multiple k-spaces to multiple images
  void reverse(Cx4 &x) const; //!< Multiple k-space to multiple images

  float scale() const; //!< Return the scaling for the unitary transform

private:
  void forwardPhase(Cx4 &x, float const scale) const;
  void reversePhase(Cx4 &x, float const scale) const;
  Cx4 &grid_;
  Cxd1 phX_, phY_, phZ_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
  bool threaded_;
};
