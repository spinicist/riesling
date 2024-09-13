#include "log.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"

using namespace rl;

void main_log(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to dump log from");
  ParseCommand(parser, iname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(iname.Get());
  if (reader.exists("log")) {
    fmt::print("{}", reader.readString("log"));
  } else {
    Log::Fail(cmd, "File {} does not contain a log", iname.Get());
  }
}
