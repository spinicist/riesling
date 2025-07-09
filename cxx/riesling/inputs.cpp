#include "inputs.hpp"

#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"

#include <algorithm>
#include <cstdlib>

using namespace rl;

template <int ND> CoreArgs<ND>::CoreArgs(args::Subparser &parser)
  : iname(parser, "FILE", "Input HD5 file")
  , oname(parser, "FILE", "Output HD5 file")
  , matrix(parser, "M", "Override matrix size", {"matrix", 'm'}, Sz<ND>())
  , basisFile(parser, "B", "Read basis from file", {"basis", 'b'})
{
}

template struct CoreArgs<2>;
template struct CoreArgs<3>;

template <int ND> GridArgs<ND>::GridArgs(args::Subparser &parser)
  : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov", 'f'}, Eigen::Array<float, ND, 1>::Zero())
  , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp", 'o'}, 1.3f)
{
}

template <int ND> auto GridArgs<ND>::Get() -> rl::GridOpts<ND>
{
  return typename rl::GridOpts<ND>{.fov = fov.Get(), .osamp = osamp.Get()};
}

template struct GridArgs<2>;
template struct GridArgs<3>;

ReconArgs::ReconArgs(args::Subparser &parser)
  : tophat(parser, "T", "Use a top hat / nearest neighbour kernel (Cartesian recon)", {"tophat"})
  , decant(parser, "D", "Direct Virtual Coil (SENSE via convolution)", {"decant"})
  , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
{
}

auto ReconArgs::Get() -> rl::ReconOpts
{
  return rl::ReconOpts{.tophat = tophat.Get(), .decant = decant.Get(), .lowmem = lowmem.Get()};
}

PreconArgs::PreconArgs(args::Subparser &parser)
  : type(parser, "P", "Pre-conditioner (none/single/multi/filename)", {"precon", 'p'}, "single")
  , λ(parser, "BIAS", "Pre-conditioner regularization (1)", {"precon-lambda"}, 1.e-3f)
{
}

auto PreconArgs::Get() -> rl::PreconOpts { return rl::PreconOpts{.type = type.Get(), .λ = λ.Get()}; }

LSMRArgs::LSMRArgs(args::Subparser &parser)
  : its(parser, "N", "Max iterations (4)", {"max-its", 'i'}, 4)
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

PDHGArgs::PDHGArgs(args::Subparser &parser)
  : its(parser, "N", "Max iterations (4)", {"max-its", 'i'}, 16)
  , resTol(parser, "A", "Tolerance on residual (1e-6)", {"res-tol", 'r'}, 1.e-6f)
  , deltaTol(parser, "B", "Tolerance on update (1e-6)", {"delta-tol", 'd'}, 1.e-6f)
  , λA(parser, "λA", "Max Eigenvalue of system matrix (1)", {"lambda-A", 'a'}, 1.f)
  , λG(parser, "λG", "Max Eigenvalue of regularizer transform (16)", {"lambda-G", 'g'}, 16.f)
{
}

auto PDHGArgs::Get() -> rl::PDHG::Opts
{
  return rl::PDHG::Opts{.imax = its.Get(), .resTol = resTol.Get(), .deltaTol = deltaTol.Get(), .λA = λA.Get(), .λG = λG.Get()};
}

ADMMArgs::ADMMArgs(args::Subparser &parser)
  : in_its0(parser, "ITS", "Initial inner iterations (64)", {"max-its0"}, 64)
  , in_its1(parser, "ITS", "Subsequent inner iterations (64)", {"max-its1"}, 64)
  , atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f)
  , btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f)
  , ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f)
  , out_its(parser, "ITS", "ADMM max iterations (64)", {"max-outer-its"}, 64)
  , ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f)
  , ε(parser, "ε", "ADMM convergence tolerance (1e-3)", {"eps"}, 1.e-3f)
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

template <int ND> SENSEArgs<ND>::SENSEArgs(args::Subparser &parser)
  : type(parser, "T", "SENSE type (auto/file.h5)", {"sense", 's'}, "auto")
  , tp(parser, "T", "SENSE calibration timepoint (first)", {"sense-tp"}, 0)
  , kWidth(parser, "K", "SENSE kernel width (10)", {"sense-width"}, 10)
  , res(parser, "R", "SENSE calibration res (6,6,6)", {"sense-res"}, Eigen::Array<float, ND, 1>::Constant(6.f))
  , l(parser, "L", "SENSE Sobolev parameter (4)", {"sense-l"}, 4.f)
  , λ(parser, "L", "SENSE Regularization (1e-4)", {"sense-lambda"}, 1.e-4f)
{
}

template <int ND> auto SENSEArgs<ND>::Get() -> rl::SENSE::Opts<ND>
{
  return rl::SENSE::Opts<ND>{
    .type = type.Get(), .tp = tp.Get(), .kWidth = kWidth.Get(), .res = res.Get(), .l = l.Get(), .λ = λ.Get()};
}

template struct SENSEArgs<2>;
template struct SENSEArgs<3>;

f0Args::f0Args(args::Subparser &parser)
  : τacq(parser, "τ", "Total ACQ time", {"tacq"})
  , Nτ(parser, "N", "Number of timesteps for f0 correction", {"Nt"})
{
}

auto f0Args::Get() -> rl::f0Opts { return rl::f0Opts{.τacq = τacq.Get(), .Nτ = Nτ.Get()}; }
