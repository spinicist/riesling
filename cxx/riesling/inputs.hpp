#pragma once

#include <args.hxx>
#include <map>
#include <optional>
#include <vector>

#include "sys/args.hpp"
#include "trajectory.hpp"
#include "types.hpp"

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

void ParseCommand(args::Subparser &parser);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname);

struct CoreOpts
{
  CoreOpts(args::Subparser &parser);
  args::Positional<std::string>                  iname, oname;
  args::ValueFlag<std::string>                   basisFile, residual;
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov;
  args::ValueFlag<rl::Sz3, SzReader<3>>          matrix;
};

struct PreconOpts
{
  PreconOpts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<float>       bias;
};

struct LsqOpts
{
  LsqOpts(args::Subparser &parser);
  args::ValueFlag<Index> its;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;
  args::ValueFlag<float> λ;
};

struct RlsqOpts
{
  RlsqOpts(args::Subparser &parser);
  args::ValueFlag<std::string> scaling;

  args::ValueFlag<Index> inner_its0;
  args::ValueFlag<Index> inner_its1;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;

  args::ValueFlag<Index> outer_its;
  args::ValueFlag<float> ρ;
  args::ValueFlag<float> ε;
  args::ValueFlag<float> μ;
  args::ValueFlag<float> τ;
};
