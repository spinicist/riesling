#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "trajectory.hpp"
#include "types.hpp"

using namespace rl;

void main_h5(args::Subparser &parser)
{
  args::Positional<std::string>    iname(parser, "FILE", "Input HD5 file to dump info from");
  args::ValueFlagList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed", {"meta", 'm'});
  args::ValueFlag<Index>           dim(parser, "D", "Print size of dimension", {"dim", 'd'}, 0);
  args::ValueFlag<std::string>     dset(parser, "D", "Dataset to interrogate (data)", {"dset"}, "data");
  args::Flag                       all(parser, "META", "Print all meta data", {"all", 'a'});

  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  if (keys) {
    auto const &meta = reader.readMeta();
    for (auto const &k : keys.Get()) {
      try {
        fmt::print("{} ", meta.at(k));
      } catch (std::out_of_range const &) {
        Log::Fail("Could not find key {}", k);
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
      auto const i = reader.readInfo();
      fmt::print("Voxel-size: {}\n", i.voxel_size.transpose());
      fmt::print("TR:         {}\n", i.tr);
      fmt::print("Origin:     {}\n", i.origin.transpose());
      fmt::print("Direction:\n{}\n", fmt::streamed(i.direction));
      if (reader.exists("trajectory")) {
        Trajectory traj(reader, i.voxel_size);
        fmt::print("Trajectory: Samples {} Traces {} Matrix {} FOV {}\n", traj.nSamples(), traj.nTraces(), traj.matrix(),
                   traj.FOV());
      }
    }

    auto const datasets = reader.list();
    if (datasets.empty()) { Log::Fail("No datasets found in {}", iname.Get()); }
    for (auto const &ds : datasets) {
      if (ds != "info" && ds != "trajectory") {
        fmt::print("Name: {:12} Shape: {:24} Names: {}\n", ds, fmt::format("{}", reader.dimensions(ds)), reader.listNames(ds));
      }
    }
  }
}
