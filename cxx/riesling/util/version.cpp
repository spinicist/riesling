#include "version.h"
#include "log.hpp"
#include "inputs.hpp"

void main_version(args::Subparser &parser)
{
  parser.Parse();
  fmt::print("Version: {}\nCompile date: {}\n", VERSION, DATETIME);
}
