#include "args.hpp"

#include "rl/log/log.hpp"

#include <exception>
#include <fmt/format.h>
#include <scn/scan.h>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{{0, Log::Display::None},
                                               {1, Log::Display::Ephemeral},
                                               {2, Log::Display::Low},
                                               {3, Log::Display::Mid},
                                               {4, Log::Display::High}};
}

args::Group                      global_group("GLOBAL OPTIONS");
args::HelpFlag                   help(global_group, "H", "Show this help message", {'h', "help"});
args::MapFlag<int, Log::Display> verbosity(global_group, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);

void SetLogging(std::string const &name)
{
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print(name, "Welcome to GEWURZ");
}

void ParseCommand(args::Subparser &parser)
{
  args::GlobalOptions globals(parser, global_group);
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
}

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }
}

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname)
{
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }
  if (!oname) { throw args::Error("No output file specified"); }
}

class ArgsError : public std::runtime_error
{
public:
  ArgsError(std::string const &msg)
    : std::runtime_error(msg)
  {
  }
};
