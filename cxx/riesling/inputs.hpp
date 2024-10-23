#pragma once

#include <args.hxx>
#include <map>
#include <optional>
#include <vector>

#include "op/grid.hpp"
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
  args::Positional<std::string> iname, oname;
  args::ValueFlag<std::string>  basisFile;
  args::Flag                    residual;
};

template <int ND> struct GridArgs
{
  ArrayFlag<float, ND>         fov;
  SzFlag<ND>                   matrix;
  args::ValueFlag<float>       osamp;
  args::ValueFlag<std::string> ktype;
  args::Flag                   vcc, lowmem;
  args::ValueFlag<Index>       subgridSize;

  GridArgs(args::Subparser &parser)
    : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov"}, Eigen::Array<float, ND, 1>::Zero())
    , matrix(parser, "M", "Grid matrix size", {"matrix", 'm'}, rl::Sz<ND>())
    , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp"}, 1.3f)
    , ktype(parser, "K", "Grid kernel - NN/KBn/ESn (ES4)", {'k', "kernel"}, "ES4")
    , vcc(parser, "V", "Virtual Conjugate Coils", {"vcc"})
    , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
    , subgridSize(parser, "B", "Subgrid size (8)", {"subgrid-size"}, 8)
  {
  }

  auto Get() -> rl::GridOpts<ND>
  {
    return rl::GridOpts{.fov = fov.Get(),
                        .matrix = matrix.Get(),
                        .osamp = osamp.Get(),
                        .ktype = ktype.Get(),
                        .vcc = vcc.Get(),
                        .lowmem = lowmem.Get(),
                        .subgridSize = subgridSize.Get()};
  }
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
