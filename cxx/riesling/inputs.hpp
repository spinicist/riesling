#pragma once

#include <args.hxx>
#include <map>
#include <optional>
#include <vector>

#include "args.hpp"
#include "op/grid.hpp"
#include "op/recon.hpp"
#include "precon.hpp"
#include "trajectory.hpp"
#include "types.hpp"

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

  GridArgs(args::Subparser &parser)
    : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov"}, Eigen::Array<float, ND, 1>::Zero())
    , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp"}, 1.3f)
    , ktype(parser, "K", "Grid kernel - NN/KBn/ESn (ES4)", {'k', "kernel"}, "ES4")
    , vcc(parser, "V", "Virtual Conjugate Coils", {"vcc"})
    , subgridSize(parser, "B", "Subgrid size (8)", {"subgrid-size"}, 8)
  {
  }

  auto Get() -> rl::TOps::Grid<ND>::Opts
  {
    return typename rl::TOps::Grid<ND>::Opts{
      .fov = fov.Get(), .osamp = osamp.Get(), .ktype = ktype.Get(), .vcc = vcc.Get(), .subgridSize = subgridSize.Get()};
  }
};

struct ReconArgs
{
  args::Flag decant, lowmem;

  ReconArgs(args::Subparser &parser)
    : decant(parser, "D", "Direct Virtual Coil (SENSE via convolution)", {"decant"})
    , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
  {
  }

  auto Get() -> rl::Recon::Opts { return rl::Recon::Opts{.decant = decant.Get(), .lowmem = lowmem.Get()}; }
};

struct PreconArgs
{
  args::ValueFlag<std::string> type;
  args::ValueFlag<float>       λ;

  PreconArgs(args::Subparser &parser)
    : type(parser, "P", "Pre-conditioner (none/single/multi/filename)", {"precon"}, "single")
    , λ(parser, "BIAS", "Pre-conditioner regularization (1)", {"precon-lambda"}, 1.f)
  {
  }

  auto Get() -> rl::PreconOpts { return rl::PreconOpts{.type = type.Get(), .λ = λ.Get()}; }
};

struct LsqArgs
{
  LsqArgs(args::Subparser &parser);
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

struct SENSEArgs
{
  args::ValueFlag<std::string> type;
  args::ValueFlag<Index>       tp, kWidth;
  ArrayFlag<float, 3>          res;
  args::ValueFlag<float>       l, λ;

  SENSEArgs(args::Subparser &parser)
    : type(parser, "T", "SENSE type (auto/file.h5)", {"sense", 's'}, "auto")
    , tp(parser, "T", "SENSE calibration timepoint (first)", {"sense-tp"}, 0)
    , kWidth(parser, "K", "SENSE kernel width (10)", {"sense-width"}, 10)
    , res(parser, "R", "SENSE calibration res (6,6,6)", {"sense-res"}, Eigen::Array3f::Constant(6.f))
    , l(parser, "L", "SENSE Sobolev parameter (4)", {"sense-l"}, 4.f)
    , λ(parser, "L", "SENSE Regularization (1e-4)", {"sense-lambda"}, 1.e-4f)
  {
  }

  auto Get() -> rl::SENSE::Opts
  {
    return rl::SENSE::Opts{
      .type = type.Get(), .tp = tp.Get(), .kWidth = kWidth.Get(), .res = res.Get(), .l = l.Get(), .λ = λ.Get()};
  }
};
