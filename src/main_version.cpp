#include "log.h"
#include "parse_args.h"
#include "version.h"

int main_version(args::Subparser &parser)
{
  parser.Parse();
  fmt::print(FMT_STRING("{}\n"), VERSION);
  return EXIT_SUCCESS;
}
