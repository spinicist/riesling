#include "io/hd5.hpp"
#include "parse_args.h"
#include "types.h"

int main_meta(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to read meta-data from");
  args::PositionalList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed");

  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const &meta = reader.readMeta();
  auto const info = reader.trajectory().info();

  for (auto const &k : keys.Get()) {
    if (meta.size() > 0) {
      for (auto const &kvp : meta) {
        if (k == kvp.first) {
          fmt::print("{}\n", kvp.second);
          break;
        }
      }
    }
    if (k == "matrix") {
      fmt::print("{}\n", info.matrix.transpose());
    } else if (k == "channels") {
      fmt::print("{}\n", info.channels);
    } else if (k == "read_points") {
      fmt::print("{}\n", info.read_points);
    } else if (k == "spokes") {
      fmt::print("{}\n", info.spokes);
    } else if (k == "volumes") {
      fmt::print("{}\n", info.volumes);
    } else if (k == "frames") {
      fmt::print("{}\n", info.frames);
    }
  }
  return EXIT_SUCCESS;
}
