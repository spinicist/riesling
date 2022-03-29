#pragma once

#include "../log.h"

template <typename Dims>
void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) {
    Log::Fail(FMT_STRING("Dimensions mismatch {} != {}"), a, b);
  }
}
