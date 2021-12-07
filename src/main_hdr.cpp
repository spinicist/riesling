#include "io.h"
#include "parse_args.h"
#include "types.h"

int main_hdr(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to print header from");
  Log log = ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get(), log);
  auto const &info = reader.readInfo();

  fmt::print(
    FMT_STRING("Header\n"
               "Matrix: {}\nVoxel-size: {}\n"
               "Read points: {} Gap: {} Spokes: {}\n"
               "Channels: {} Volumes: {} Echoes: {}\n"
               "TR: {}\nOrigin: {}\nDirection:\n{}\n"),
    info.matrix.transpose(),
    info.voxel_size.transpose(),
    info.read_points,
    info.read_gap,
    info.spokes,
    info.channels,
    info.volumes,
    info.echoes,
    info.tr,
    info.origin.transpose(),
    info.direction);
  auto const &meta = reader.readMeta();
  if (meta.size() > 0) {
    fmt::print("Meta data:\n");
    for (auto const &kvp : meta) {
      fmt::print("{}: {}\n", kvp.first, kvp.second);
    }
  }

  return EXIT_SUCCESS;
}
