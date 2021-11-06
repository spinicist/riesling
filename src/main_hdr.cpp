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
    "Header\n"
    "Matrix: {}\nVoxel-size: {}\n"
    "Read points: {} Gap: {}\n"
    "Hi-res spokes: {} Lo-res spokes: {} Lo-res scale: {}\n"
    "Channels: {} Volumes: {} TR: {}\n"
    "Origin: {}\nDirection:\n{}\n",
    info.matrix.transpose(),
    info.voxel_size.transpose(),
    info.read_points,
    info.read_gap,
    info.spokes_hi,
    info.spokes_lo,
    info.lo_scale,
    info.channels,
    info.volumes,
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
