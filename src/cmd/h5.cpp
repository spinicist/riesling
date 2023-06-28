#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.hpp"

using namespace rl;

int main_h5(args::Subparser &parser)
{
  args::Positional<std::string>    iname(parser, "FILE", "Input HD5 file to dump info from");
  args::Flag                       info(parser, "INFO", "Print header information", {"info", 'i'});
  args::Flag                       dsets(parser, "DSETS", "List datasets and dimensions", {"dsets", 'd'});
  args::ValueFlagList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed", {"meta", 'm'});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  if (info) {
    auto const i = reader.readInfo();
    fmt::print(
      "Matrix:     {}\n"
      "Voxel-size: {}\n"
      "TR:         {}\n"
      "Origin: {}\n"
      "Direction:\n{}\n",
      i.matrix,
      i.voxel_size.transpose(),
      i.tr,
      i.origin.transpose(),
      i.direction);
  } else if (dsets) {
    auto const datasets = reader.list();
    if (datasets.empty()) { Log::Fail("No datasets found in {}", iname.Get()); }
    for (auto const &ds : datasets) {
      fmt::print("{} ", ds);
      switch (reader.rank(ds)) {
      case 1: fmt::print("{}\n", reader.dimensions<1>(ds)); break;
      case 2: fmt::print("{}\n", reader.dimensions<2>(ds)); break;
      case 3: fmt::print("{}\n", reader.dimensions<3>(ds)); break;
      case 4: fmt::print("{}\n", reader.dimensions<4>(ds)); break;
      case 5: fmt::print("{}\n", reader.dimensions<5>(ds)); break;
      case 6: fmt::print("{}\n", reader.dimensions<6>(ds)); break;
      default: fmt::print("rank is higher than 6\n");
      }
    }
  } else if (keys) {
    auto const &meta = reader.readMeta();
    for (auto const &k : keys.Get()) {
      try {
        fmt::print("{}\n", meta.at(k));
      } catch (std::out_of_range const &) {
        Log::Fail("Could not find key {}", k);
      }
    }
  }
  return EXIT_SUCCESS;
}
