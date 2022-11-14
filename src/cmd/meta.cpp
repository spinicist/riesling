#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "types.hpp"

using namespace rl;

int main_meta(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to read meta-data from");
  args::PositionalList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed");
  args::Flag all(parser, "A", "Print all meta-data", {"all"});

  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  auto const &meta = reader.readMeta();
  Trajectory traj(reader);

  if (all) {
    for (auto const &m: meta) {
      fmt::print("{}: {}\n", m.first, m.second);
    }
  } else {
    for (auto const &k : keys.Get()) {
      if (k == "matrix") {
        fmt::print("{}\n", traj.info().matrix);
        continue;
      } else if (k == "samples") {
        fmt::print("{}\n", traj.nSamples());
        continue;
      } else if (k == "traces") {
        fmt::print("{}\n", traj.nTraces());
        continue;
      }

      if (meta.size() > 0) {
        try {
          fmt::print("{}\n", meta.at(k));
        } catch (std::out_of_range const &) {
          Log::Fail("Could not find key {}", k);
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
