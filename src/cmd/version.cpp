#include "version.h"
#include "log.h"
#include "parse_args.hpp"

int main_version(args::Subparser &parser)
{
  parser.Parse();
  fmt::print(FMT_STRING("Version: {}\nCompile date: {}\n"), VERSION, DATETIME);
  return EXIT_SUCCESS;
}
