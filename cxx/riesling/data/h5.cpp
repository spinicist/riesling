#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_h5(args::Subparser &parser)
{
  args::Positional<std::string>    iname(parser, "FILE", "Input HD5 file to dump info from");
  args::ValueFlagList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed", {"meta", 'm'});
  args::ValueFlag<Index>           dim(parser, "D", "Print size of dimension", {"dim", 'd'}, 0);
  args::ValueFlag<std::string>     dset(parser, "D", "Dataset to interrogate (data)", {"dset"}, "data");
  args::Flag                       all(parser, "META", "Print all meta data", {"all", 'a'});

  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());

  if (keys) {
    auto const &meta = reader.readMeta();
    for (auto const &k : keys.Get()) {
      try {
        fmt::print("{} ", meta.at(k));
      } catch (std::out_of_range const &) {
        throw Log::Failure(cmd, "Could not find key {}", k);
      }
    }
    fmt::print("\n");
  } else if (all) {
    auto const &meta = reader.readMeta();
    for (auto const &kvp : meta) {
      fmt::print("{}: {}\n", kvp.first, kvp.second);
    }
  } else if (dim) {
    fmt::print("{}\n", reader.dimensions(dset.Get()).at(dim.Get()));
  } else {
    if (reader.exists("info")) {
      auto const i = reader.readStruct<Info>(HD5::Keys::Info);
      fmt::print("Voxel-size: {}\n", i.voxel_size.transpose());
      fmt::print("TR:         {}\n", i.tr);
      fmt::print("Origin:     {}\n", i.origin.transpose());
      fmt::print("Direction:\n{}\n", fmt::streamed(i.direction));
      if (reader.exists("trajectory")) {
        auto const td = reader.dimensions(HD5::Keys::Trajectory)[0];
        switch (td) {
        case 2: {
          TrajectoryN<2> traj(reader, i.voxel_size.head(2));
          fmt::print("Trajectory: Samples {} Traces {} Matrix {} FOV {}\n", traj.nSamples(), traj.nTraces(), traj.matrix(),
                     traj.FOV());
        } break;
        case 3: {
          TrajectoryN<3> traj(reader, i.voxel_size);
          fmt::print("Trajectory: Samples {} Traces {} Matrix {} FOV {}\n", traj.nSamples(), traj.nTraces(), traj.matrix(),
                     traj.FOV());
        } break;
        default: throw(Log::Failure(cmd, "Unknown trajectory dimension {}", td));
        }
      }
    }

    auto const datasets = reader.list();
    if (datasets.empty()) { throw Log::Failure(cmd, "No datasets found in {}", iname.Get()); }
    for (auto const &ds : datasets) {
      if (ds != "info" && ds != "trajectory" && ds != "meta") {
        fmt::print("Name: {:12} Shape: {:24} Names: {}\n", ds, fmt::format("{}", reader.dimensions(ds)), reader.listNames(ds));
      }
    }
  }
  Log::Print(cmd, "Finished");
}
