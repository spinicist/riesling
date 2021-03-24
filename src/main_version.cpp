#include "log.h"
#include "parse_args.h"
#include "version.h"

int main_version(args::Subparser &parser)
{
  parser.Parse();
  Log::Level const level = Log::Level::Info;
  Log log(level);

  log.info(FMT_STRING("{}"), VERSION);

  return EXIT_SUCCESS;
}
