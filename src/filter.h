#pragma once

#include "radial.h"

#include "log.h"
#include <functional>

Eigen::ArrayXf
RadialTukey(Eigen::Index const n, long const start_n, long const end_n, float const ea);

float Tukey(float const &r, float const &flat_width, float const &end, float const &end_height);
void ImageFilter(std::function<float(float const &)> const &f, Cx3 &x, Log &log);
void KSFilter(std::function<float(float const &)> const &f, Cx4 &x, Log &log);
void ImageTukey(float const &start_r, float const &end_r, float const &end_h, Cx3 &x, Log &log);
void KSTukey(float const &start_r, float const &end_r, float const &end_h, Cx4 &x, Log &log);