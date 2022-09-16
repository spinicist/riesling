#pragma once

#include "log.hpp"
#include "types.hpp"

namespace rl {
Cx4 llr_sliding(Cx4 const &x, float const l, Index const kSz);
Cx4 llr_patch(Cx4 const &x, float const l, Index const kSz);
}
