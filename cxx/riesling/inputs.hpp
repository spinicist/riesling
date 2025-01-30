#pragma once

#include "args.hpp"

#include "rl/algo/admm.hpp"
#include "rl/algo/lsmr.hpp"
#include "rl/op/grid-opts.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

#include <map>
#include <optional>
#include <vector>

struct CoreArgs
{
  args::Positional<std::string> iname, oname;
  SzFlag<3>                     matrix;
  args::ValueFlag<std::string>  basisFile;
  args::Flag                    residual;

  CoreArgs(args::Subparser &parser);
};

template <int ND> struct GridArgs
{
  ArrayFlag<float, ND>   fov;
  args::ValueFlag<float> osamp;

  GridArgs(args::Subparser &parser);
  auto Get() -> rl::GridOpts<ND>;
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
  args::ValueFlag<Index> its;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;
  args::ValueFlag<float> λ;

  LSMRArgs(args::Subparser &parser);
  auto Get() -> rl::LSMR::Opts;
};

struct ADMMArgs
{
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

  ADMMArgs(args::Subparser &parser);
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
