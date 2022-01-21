#pragma once

#include "types.h" // Make sure to include this first to set EIGEN_USE_THREADS

#include "fftw3.h"
#include "threads.h"

namespace FFT {

void Start();
void End();
void SetTimelimit(double time);
Cx1 Phase(Index const sz);

} // namespace FFT
