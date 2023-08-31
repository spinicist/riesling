#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.hpp"

using namespace rl;

int main_h5(args::Subparser &parser)
{
  args::Positional<std::string>    iname(parser, "FILE", "Input HD5 file to dump info from");
  args::ValueFlagList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed", {"meta", 'm'});
  args::Flag                       all(parser, "META", "Print all meta data", {"all", 'a'});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  if (keys) {
    auto const &meta = reader.readMeta();
    for (auto const &k : keys.Get()) {
      try {
        fmt::print("{}\n", meta.at(k));
      } catch (std::out_of_range const &) {
        Log::Fail("Could not find key {}", k);
      }
    }
  } else if (all) {
    auto const &meta = reader.readMeta();
    for (auto const &kvp : meta) {
      fmt::print("{}: {}\n", kvp.first, kvp.second);
    }
  } else {
    if (reader.exists("info")) {
      auto const i = reader.readInfo();
      fmt::print("Matrix:     {}\n"
                 "Voxel-size: {}\n"
                 "TR:         {}\n"
                 "Origin:     {}\n"
                 "Direction:\n{}\n",
                 i.matrix, i.voxel_size.transpose(), i.tr, i.origin.transpose(), i.direction);
    }

    auto const datasets = reader.list();
    if (datasets.empty()) { Log::Fail("No datasets found in {}", iname.Get()); }
    for (auto const &ds : datasets) {
      if (ds != "info") { fmt::print("{}:{}\n", ds, fmt::join(reader.dimensions(ds), ",")); }
    }
  }
  return EXIT_SUCCESS;
}
