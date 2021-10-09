#pragma once

#include "log.h"
#include "types.h"

Cx4 llr(Cx4 const &x, float const l, long const kSz, Log &log);
Cx4 llr_patch(Cx4 const &x, float const l, long const kSz, Log &log);
