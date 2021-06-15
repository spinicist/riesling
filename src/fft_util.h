#pragma once

#include "types.h" // Make sure to include this first to set EIGEN_USE_THREADS

#include "fftw3.h"
#include "log.h"
#include "threads.h"

namespace FFT {

void Start(Log &log);
void End(Log &log);
void SetTimelimit(double time);
Cx1 Phase(long const sz);

} // namespace FFT
