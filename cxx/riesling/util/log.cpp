#include "log.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"

using namespace rl;

void main_log(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to dump log from");
  args::ValueFlag<std::string>  filter(parser, "F", "Filter category", {"filter", 'f'});
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  if (reader.exists("log")) {
    auto const entries = reader.readStrings("log");
    for (auto const &entry : entries) {
      if (!filter || (entry.substr(12, 6).find(filter.Get())) != std::string::npos) { fmt::print("{}\n", entry); }
    }
  } else {
    Log::Fail(cmd, "File {} does not contain a log", iname.Get());
  }
  Log::Print(cmd, "Finished");
}
