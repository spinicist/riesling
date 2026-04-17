#pragma once

#include "args.hpp"

#include "rl/algo/admm.hpp"

struct ADMMArgs
{
  args::ValueFlag<Index> in_its0;
  args::ValueFlag<Index> in_its1;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;

  args::ValueFlag<float> ε;
  args::ValueFlag<Index> outerIts;
  args::ValueFlag<Index> restart;

  args::ValueFlag<float> ρ;
  args::Flag             updateρ;
  args::ValueFlag<float> τ;

  ADMMArgs(args::Subparser &parser);
  auto Get() -> rl::ADMM::Opts;
};
