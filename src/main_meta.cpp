#include "io.h"
#include "parse_args.h"
#include "types.h"

int main_meta(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file to read meta-data from");
  args::PositionalList<std::string> keys(parser, "KEYS", "Meta-data keys to be printed");

  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const &meta = reader.readMeta();
  if (meta.size() > 0) {
    for (auto const &k : keys.Get()) {
      for (auto const &kvp : meta) {
        if (k == kvp.first) {
          fmt::print("{}\n", kvp.second);
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
