#pragma once

#include "info.h"

#include "log.h"
#include <functional>

void ImageTukey(float const &start_r, float const &end_r, float const &end_h, Cx3 &x, Log &log);
void KSTukey(float const &start_r, float const &end_r, float const &end_h, Cx4 &x, Log &log);