#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.h"

using namespace rl;

int main_hdr(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to print header from");
  args::Flag printMeta(parser, "M", "Print meta-data", {"meta", 'm'});
  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();

  fmt::print(
    FMT_STRING("Trajectory Samples: {} Traces: {} Frames: {}\n"
               "Matrix: {}\n"
               "Voxel-size: {}\t TR: {}\t Origin: {}\n"
               "Direction:\n{}\n"),
    traj.nSamples(),
    traj.nTraces(),
    traj.nFrames(),
    info.matrix,
    info.voxel_size.transpose(),
    info.tr,
    info.origin.transpose(),
    info.direction);
  if (printMeta) {
    auto const &meta = reader.readMeta();
    if (meta.size() > 0) {
      fmt::print("Meta data:\n");
      for (auto const &kvp : meta) {
        fmt::print("{}: {}\n", kvp.first, kvp.second);
      }
    }
  }

  return EXIT_SUCCESS;
}
