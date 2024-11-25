#include "inputs.hpp"

#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"

#include <algorithm>
#include <cstdlib>

using namespace rl;

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None}, {1, Log::Level::Ephemeral}, {2, Log::Level::Standard}, {3, Log::Level::Debug}};
}

CoreArgs::CoreArgs(args::Subparser &parser)
  : iname(parser, "FILE", "Input HD5 file")
  , oname(parser, "FILE", "Output HD5 file")
  , matrix(parser, "M", "Override matrix size", {"matrix", 'm'}, Sz3())
  , basisFile(parser, "B", "Read basis from file", {"basis", 'b'})
  , residual(parser, "R", "Write out residual to file", {"residual", 'r'})
{
}

LsqArgs::LsqArgs(args::Subparser &parser)
  : its(parser, "N", "Max iterations (4)", {'i', "max-its"}, 4)
  , atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f)
  , λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f)
{
}

RlsqOpts::RlsqOpts(args::Subparser &parser)
  : scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu")
  , inner_its0(parser, "ITS", "Initial inner iterations (4)", {"max-its0"}, 4)
  , inner_its1(parser, "ITS", "Subsequent inner iterations (1)", {"max-its"}, 1)
  , atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f)
  , outer_its(parser, "ITS", "ADMM max iterations (20)", {"max-outer-its"}, 20)
  , ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f)
  , ε(parser, "ε", "ADMM convergence tolerance (1e-2)", {"eps"}, 1.e-2f)
  , μ(parser, "μ", "ADMM residual rescaling tolerance (default 1.2)", {"mu"}, 1.2f)
  , τ(parser, "τ", "ADMM residual rescaling maximum (default 10)", {"tau"}, 10.f)
{
}

args::Group    global_group("GLOBAL OPTIONS");
args::HelpFlag help(global_group, "H", "Show this help message", {'h', "help"});
args::MapFlag<int, Log::Level>
                             verbosity(global_group, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Level::Standard);
args::ValueFlag<std::string> debug(global_group, "F", "Write debug images to file", {"debug"});
args::ValueFlag<Index>       nthreads(global_group, "N", "Limit number of threads", {"nthreads"});

void SetLogging(std::string const &name)
{
  if (verbosity) {
    Log::SetLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print(name, "Welcome to RIESLING");
  if (debug) { Log::SetDebugFile(debug.Get()); }
}

void SetThreadCount()
{
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads.Get());
  } else if (char *const env_p = std::getenv("RL_THREADS")) {
    Threads::SetGlobalThreadCount(std::atoi(env_p));
  }
}

void ParseCommand(args::Subparser &parser)
{
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
  SetThreadCount();
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
