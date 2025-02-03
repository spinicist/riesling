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

template <int ND>
GridArgs<ND>::GridArgs(args::Subparser &parser)
  : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov"}, Eigen::Array<float, ND, 1>::Zero())
  , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp"}, 1.3f)
  , ktype(parser, "K", "Grid kernel - NN/KBn/ESn (ES4)", {'k', "kernel"}, "ES4")
  , subgridSize(parser, "B", "Subgrid size (8)", {"subgrid-size"}, 8)
{
}

template <int ND> auto GridArgs<ND>::Get() -> rl::TOps::Grid<ND>::Opts
{
  return typename rl::TOps::Grid<ND>::Opts{
    .fov = fov.Get(), .osamp = osamp.Get(), .ktype = ktype.Get(), .subgridSize = subgridSize.Get()};
}

template struct GridArgs<2>;
template struct GridArgs<3>;

ReconArgs::ReconArgs(args::Subparser &parser)
  : decant(parser, "D", "Direct Virtual Coil (SENSE via convolution)", {"decant"})
  , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
{
}

auto ReconArgs::Get() -> rl::Recon::Opts { return rl::Recon::Opts{.decant = decant.Get(), .lowmem = lowmem.Get()}; }

PreconArgs::PreconArgs(args::Subparser &parser)
  : type(parser, "P", "Pre-conditioner (none/single/multi/filename)", {"precon"}, "single")
  , λ(parser, "BIAS", "Pre-conditioner regularization (1)", {"precon-lambda"}, 1.e-3f)
{
}

auto PreconArgs::Get() -> rl::PreconOpts { return rl::PreconOpts{.type = type.Get(), .λ = λ.Get()}; }

LSMRArgs::LSMRArgs(args::Subparser &parser)
  : its(parser, "N", "Max iterations (4)", {'i', "max-its"}, 4)
  , atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f)
  , λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f)
{
}

auto LSMRArgs::Get() -> rl::LSMR::Opts
{
  return rl::LSMR::Opts{.imax = its.Get(), .aTol = atol.Get(), .bTol = btol.Get(), .cTol = ctol.Get(), .λ = λ.Get()};
}

ADMMArgs::ADMMArgs(args::Subparser &parser)
  : in_its0(parser, "ITS", "Initial inner iterations (4)", {"max-its0"}, 4)
  , in_its1(parser, "ITS", "Subsequent inner iterations (1)", {"max-its"}, 1)
  , atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f)
  , out_its(parser, "ITS", "ADMM max iterations (20)", {"max-outer-its"}, 20)
  , ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f)
  , ε(parser, "ε", "ADMM convergence tolerance (1e-2)", {"eps"}, 1.e-2f)
  , μ(parser, "μ", "Residual balancing tolerance (default 1.2)", {"mu"}, 1.2f)
  , τ(parser, "τ", "Residual balancing ratio limit (default 10)", {"tau"}, 10.f)
  , ɑ(parser, "ɑ", "Over-relaxation parameter (choose 1<ɑ<2)", {"alpha"}, 0.f)
{
}

auto ADMMArgs::Get() -> rl::ADMM::Opts
{
  return rl::ADMM::Opts{.iters0 = in_its0.Get(),
                        .iters1 = in_its1.Get(),
                        .aTol = atol.Get(),
                        .bTol = btol.Get(),
                        .cTol = ctol.Get(),
                        .outerLimit = out_its.Get(),
                        .ε = ε.Get(),
                        .ρ = ρ.Get(),
                        .balance = !ρ,
                        .μ = μ.Get(),
                        .τmax = τ.Get(),
                        .ɑ = ɑ.Get()};
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
