#pragma once

#include "types.h" // Make sure to include this first to set EIGEN_USE_THREADS

#include "fftw3.h"
#include "log.h"

void FFTStart(Log &log);
void FFTEnd(Log &log);

/* 3D Fourier transform
 *
 */
struct FFT3
{
  FFT3(Cx3 &grid, Log &log);
  ~FFT3();

  void forward() const; //!< Multiple images to multiple K-spaces
  void reverse() const; //!< Multiple K-spaces to multiple images
  void shift() const;   //!< Perform multiple shifts

private:
  Cx3 &grid_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
};

/* Multiple 3D Fourier transforms executed simultaneously
 *
 */
struct FFT3N
{
  FFT3N(Cx4 &grid, Log &log);
  ~FFT3N();

  void forward() const; //!< Multiple images to multiple K-spaces
  void reverse() const; //!< Multiple K-spaces to multiple images
  void shift() const;   //!< Perform multiple shifts

private:
  Cx4 &grid_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Log &log_;
};

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