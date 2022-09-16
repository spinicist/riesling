#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.hpp"

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
    fmt::print("{} ", ds);
    switch (reader.rank(ds)) {
    case 1:
      fmt::print("{}\n", reader.dimensions<1>(ds));
      break;
    case 2:
      fmt::print("{}\n", reader.dimensions<2>(ds));
      break;
    case 3:
      fmt::print("{}\n", reader.dimensions<3>(ds));
      break;
    case 4:
      fmt::print("{}\n", reader.dimensions<4>(ds));
      break;
    case 5:
      fmt::print("{}\n", reader.dimensions<5>(ds));
      break;
    case 6:
      fmt::print("{}\n", reader.dimensions<6>(ds));
      break;
    default:
      fmt::print("rank is higher than 6\n");
    }
  }
  return EXIT_SUCCESS;
}
