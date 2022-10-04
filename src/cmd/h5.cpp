#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.hpp"

using namespace rl;

int main_h5(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to dump info from");
  args::Flag info(parser, "INFO", "Print header information", {"info", 'i'});
  args::Flag dsets(parser, "DSETS", "List datasets and dimensions", {"dsets", 'd'});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  if (info) {
    Trajectory traj(reader);
    fmt::print(
      FMT_STRING("Trajectory Samples: {} Traces: {} Frames: {}\n"
                 "Matrix: {}\n"
                 "Voxel-size: {}\t TR: {}\t Origin: {}\n"
                 "Direction:\n{}\n"),
      traj.nSamples(),
      traj.nTraces(),
      traj.nFrames(),
      traj.info().matrix,
      traj.info().voxel_size.transpose(),
      traj.info().tr,
      traj.info().origin.transpose(),
      traj.info().direction);
  } else if (dsets) {
    auto const datasets = reader.list();
    if (datasets.empty()) {
      Log::Fail("No datasets found in {}", iname.Get());
    }
    for (auto const &ds : datasets) {
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
  }
  return EXIT_SUCCESS;
}
