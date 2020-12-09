#pragma once

#include "log.h"
#include "radial.h"

void slab_correct(RadialInfo const &info, float const pw_us, float const rbw, Cx3 &ks, Log &log);