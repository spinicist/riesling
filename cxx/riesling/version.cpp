#include "version.h"
#include "log.hpp"
#include "parse_args.hpp"

int main_version(args::Subparser &parser)
{
  parser.Parse();
  fmt::print("Version: {}\nCompile date: {}\n", VERSION, DATETIME);
  return EXIT_SUCCESS;
}
