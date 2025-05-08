#include "version.h"
#include "inputs.hpp"

#include "rl/log/log.hpp"

void main_version(args::Subparser &parser)
{
  parser.Parse();
  fmt::print("Version: {}\nCompile date: {}\n", VERSION, DATETIME);
}
