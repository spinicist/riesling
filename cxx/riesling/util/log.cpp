#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"

using namespace rl;

void main_log(args::Subparser &parser)
{
  args::Positional<std::string>    iname(parser, "FILE", "HD5 file to dump log from");
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  if (reader.exists("log")) {
    fmt::print("{}", reader.readString("log"));
  } else {
    Log::Fail("File {} does not contain a log", iname.Get());
  }
}
