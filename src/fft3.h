#pragma once

#include "fft_util.h"

/* 3D Fourier transform
 *
 */
struct FFT3
{
  FFT3(
      Sz3 const &sz,
      Log &log,
      long const nThreads = Threads::GlobalThreadCount()); //!< Will allocate working space
  FFT3(Cx3 &grid, Log &log, long const nThreads = Threads::GlobalThreadCount());
  ~FFT3();

  void forward(Cx3 &x) const; //!< Multiple images to multiple K-spaces
  void reverse(Cx3 &x) const; //!< Multiple K-spaces to multiple images

private:
  void init(Cx3 &workspace, long const nThreads);
  void forwardPhase(Cx3 &x, float const scale) const;
  void reversePhase(Cx3 &x, float const scale) const;
  Cxd1 phX_, phY_, phZ_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log log_;
  bool threaded_;
};
