#pragma once

#include "args.hpp"

#include "rl/algo/admm.hpp"
#include "rl/algo/lsmr.hpp"
#include "rl/algo/pdhg.hpp"
#include "rl/op/grid-opts.hpp"
#include "rl/op/recon-opts.hpp"
#include "rl/precon-opts.hpp"
#include "rl/sense/sense.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

#include <map>
#include <optional>
#include <vector>

template <int ND> struct CoreArgs
{
  args::Positional<std::string> iname, oname;
  SzFlag<ND>                    matrix;
  args::ValueFlag<std::string>  basisFile;

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
  auto Get() -> rl::ReconOpts;
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

struct PDHGArgs
{
  args::ValueFlag<Index> its;
  args::ValueFlag<float> resTol, deltaTol;
  args::ValueFlag<float> λA, λG;

  PDHGArgs(args::Subparser &parser);
  auto Get() -> rl::PDHG::Opts;
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

template <int ND> struct SENSEArgs
{
  args::ValueFlag<std::string> type;
  args::ValueFlag<Index>       tp, kWidth;
  ArrayFlag<float, ND>         res;
  args::ValueFlag<float>       l, λ;

  SENSEArgs(args::Subparser &parser);
  auto Get() -> rl::SENSE::Opts<ND>;
};

struct f0Args
{
  args::ValueFlag<float> τacq;
  args::ValueFlag<Index> Nτ;

  f0Args(args::Subparser &parser);
  auto Get() -> rl::f0Opts;
};
