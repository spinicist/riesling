#include "parse_args.h"
#include "log.h"
#include "version.h"

int main_version(args::Subparser &parser)
{
  parser.Parse();
  Log::Level const level = Log::Level::Info;
  Log log(level);

  log.info(FMT_STRING("v{}"), PROJECT_VER);

  return EXIT_SUCCESS;
}
