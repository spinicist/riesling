#pragma once

#include "../log.h"

namespace rl {

template <typename Dims>
void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) {
    Log::Fail(FMT_STRING("Dimensions mismatch {} != {}"), fmt::streamed(a), fmt::streamed(b));
  }
}

}