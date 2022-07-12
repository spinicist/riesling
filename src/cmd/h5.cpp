#include "io/hd5.hpp"
#include "parse_args.h"
#include "types.h"

using namespace rl;

int main_h5(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to dump info from");

  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  auto const dsets = reader.list();
  if (dsets.empty()) {
    Log::Fail("No datasets found in {}", iname.Get());
  }
  for (auto const &ds : dsets) {
    fmt::print("{}\n", ds);
  }
  return EXIT_SUCCESS;
}
