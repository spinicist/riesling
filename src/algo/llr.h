#pragma once

#include "log.h"
#include "types.h"

Cx4 llr_sliding(Cx4 const &x, float const l, Index const kSz);
Cx4 llr_patch(Cx4 const &x, float const l, Index const kSz);
