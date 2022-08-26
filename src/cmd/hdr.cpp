#include "io/hd5.hpp"
#include "parse_args.h"
#include "types.h"

using namespace rl;

int main_hdr(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to print header from");
  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const &info = reader.trajectory().info();

  fmt::print(
    FMT_STRING("Channels: {} Samples: {} Traces: {}\n"
               "Matrix: {}\n"
               "3D Gridding: {} 3D FFT: {}\n"
               "Frames: {} Volumes: {}\n"
               "Voxel-size: {}\t TR: {}\t Origin: {}\n"
               "Direction:\n{}\n"),
    info.channels,
    info.samples,
    info.traces,
    info.matrix,
    info.grid3D, info.fft3D,
    info.frames,
    info.volumes,
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
