#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <unordered_map>

enum class SDC
{
  None = 0,
  Analytic = 1,
  Pipe = 2
};

extern std::unordered_map<std::string, SDC> SDCMap;

// Forward declare
struct Gridder;
struct Kernel;

Cx2 SDCPipe(Info const &info, Gridder *gridder, Kernel *kernel, Log &log);