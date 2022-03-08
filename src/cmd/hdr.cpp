#include "io/io.h"
#include "parse_args.h"
#include "types.h"

int main_hdr(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to print header from");
  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const &info = reader.trajectory().info();

  fmt::print(
    FMT_STRING("Type: {}\n"
               "Matrix: {}\n"
               "Channels: {} Read points: {} Spokes: {}\n"
               "Volumes: {} Frames: {}\n"
               "Voxel-size: {}\t TR: {}\t Origin: {}\n"
               "Direction:\n{}\n"),
    (info.type == Info::Type::ThreeD ? "3D" : "3D Stack"),
    info.matrix.transpose(),
    info.channels,
    info.read_points,
    info.spokes,
    info.volumes,
    info.frames,
    info.voxel_size.transpose(),
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
