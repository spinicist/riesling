#pragma once

#include "args.hpp"

#include "rl/algo/admm.hpp"
#include "rl/algo/lsmr.hpp"
#include "rl/op/grid.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

#include <map>
#include <optional>
#include <vector>

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

void ParseCommand(args::Subparser &parser);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname);

struct CoreArgs
{
  CoreArgs(args::Subparser &parser);
  args::Positional<std::string> iname, oname;
  SzFlag<3>                     matrix;
  args::ValueFlag<std::string>  basisFile;
  args::Flag                    residual;
};

template <int ND> struct GridArgs
{
  ArrayFlag<float, ND>         fov;
  args::ValueFlag<float>       osamp;
  args::ValueFlag<std::string> ktype;
  args::Flag                   vcc;
  args::ValueFlag<Index>       subgridSize;
  GridArgs(args::Subparser &parser);
  auto Get() -> rl::TOps::Grid<ND>::Opts;
};

struct ReconArgs
{
  args::Flag decant, lowmem;
  ReconArgs(args::Subparser &parser);
  auto Get() -> rl::Recon::Opts;
};

struct PreconArgs
{
  args::ValueFlag<std::string> type;
  args::ValueFlag<float>       λ;
  PreconArgs(args::Subparser &parser);
  auto Get() -> rl::PreconOpts;
};

struct LSMRArgs
{
  LSMRArgs(args::Subparser &parser);
  args::ValueFlag<Index> its;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;
  args::ValueFlag<float> λ;
  auto                   Get() -> rl::LSMR::Opts;
};

struct ADMMArgs
{
  ADMMArgs(args::Subparser &parser);

  args::ValueFlag<Index> in_its0;
  args::ValueFlag<Index> in_its1;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;

  args::ValueFlag<Index> out_its;
  args::ValueFlag<float> ρ;
  args::ValueFlag<float> ε;

  args::ValueFlag<float> μ;
  args::ValueFlag<float> τ;

  args::ValueFlag<float> ɑ;

  auto Get() -> rl::ADMM::Opts;
};

struct SENSEArgs
{
  args::ValueFlag<std::string> type;
  args::ValueFlag<Index>       tp, kWidth;
  ArrayFlag<float, 3>          res;
  args::ValueFlag<float>       l, λ;

  SENSEArgs(args::Subparser &parser);
  auto Get() -> rl::SENSE::Opts;
};

struct f0Args
{
  args::ValueFlag<std::string> fname;
  VectorFlag<float>            τ;

  f0Args(args::Subparser &parser);
  auto Get() -> rl::Recon::f0Opts;
};
