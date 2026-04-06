#include "all.hpp"

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
  , dset(parser, "D", "Dataset name", {"dset", 'd'}, HD5::Keys::Data)
  , basisFile(parser, "B", "Read basis from file", {"basis", 'b'})
  , matrix(parser, "M", "Override matrix size", {"matrix", 'm'}, Sz<ND>())
  , residual(parser, "R", "Output k-space residual", {"resid", 'r'})
{
}

template struct CoreArgs<2>;
template struct CoreArgs<3>;

template <int ND> GridArgs<ND>::GridArgs(args::Subparser &parser)
  : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov", 'f'}, Eigen::Array<float, ND, 1>::Zero())
  , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp", 'o'}, 1.3f)
  , tophat(parser, "T", "Use a top hat (NN) kernel", {"tophat"})
  , kW(parser, "W", "ExpSemi kernel width", {"kwidth"}, 4)
{
}

template <int ND> auto GridArgs<ND>::Get() -> rl::GridOpts<ND>
{
  return typename rl::GridOpts<ND>{.fov = fov.Get(), .osamp = osamp.Get(), .tophat = tophat.Get(), .kW = kW.Get()};
}

template struct GridArgs<2>;
template struct GridArgs<3>;

ReconArgs::ReconArgs(args::Subparser &parser)
  : decant(parser, "D", "Direct Virtual Coil (SENSE via convolution)", {"decant"})
  , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
{
}

auto ReconArgs::Get() -> rl::ReconOpts { return rl::ReconOpts{.decant = decant.Get(), .lowmem = lowmem.Get()}; }

PreconArgs::PreconArgs(args::Subparser &parser)
  : type(parser, "P", "Preconditioner (none/single/multi/filename)", {"precon", 'p'}, "single")
  , λ(parser, "λ", "Preconditioner regularization (0)", {"precon-l"}, 0.f)
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
  : adaptive(parser, "A", "Adaptive step sizes", {"adaptive"})
  , lad(parser, "L", "Least Absolute Deviations, PDHG only", {"lad", 'l'})
  , its(parser, "N", "Max iterations (4)", {"max-its", 'i'}, 128)
  , tol(parser, "B", "PDHG residual/Δ tolerance (1e-2)", {"pdhg-tol", 't'}, 1.e-4f)
  , λE(parser, "λE", "Max Eigenvalue of encoding operator (1)", {"lambda-E", 'e'}, 1.f)
{
}

auto PDHGArgs::Get() -> rl::PDHG::Opts
{
  return rl::PDHG::Opts{.adaptive = adaptive, .lad = lad, .imax = its.Get(), .tol = tol.Get(), .λE = λE.Get()};
}

template <int ND> SENSEArgs<ND>::SENSEArgs(args::Subparser &parser)
  : type(parser, "T", "SENSE type (auto/file.h5)", {"sense", 's'}, "auto")
  , tp(parser, "T", "SENSE calibration timepoint (first)", {"sense-tp"}, 0)
  , kWidth(parser, "K", "SENSE kernel width (10)", {"sense-width"}, 10)
  , its(parser, "I", "SENSE recon max iterations (8)", {"sense-its"}, 8)
  , res(parser, "R", "SENSE calibration res (6,6,6)", {"sense-res"}, Eigen::Array<float, ND, 1>::Constant(6.f))
  , l(parser, "L", "SENSE Sobolev parameter (4)", {"sense-l"}, 4.f)
  , λ(parser, "L", "SENSE Regularization (1e-4)", {"sense-lambda"}, 1.e-4f)
  , renorm(parser, "N", "SENSE Renormalization (RSS/none)", {"sense-renorm"}, SENSE::NormMap, SENSE::Normalization::RSS)
{
}

template <int ND> auto SENSEArgs<ND>::Get() -> rl::SENSE::Opts<ND>
{
  return rl::SENSE::Opts<ND>{.type = type.Get(),
                             .tp = tp.Get(),
                             .kWidth = kWidth.Get(),
                             .its = its.Get(),
                             .res = res.Get(),
                             .l = l.Get(),
                             .λ = λ.Get(),
                             .renorm = renorm.Get()};
}

template struct SENSEArgs<2>;
template struct SENSEArgs<3>;

f0Args::f0Args(args::Subparser &parser)
  : τ0(parser, "τ", "ACQ start time", {"t0"})
  , τacq(parser, "τ", "Total ACQ time", {"tacq"})
  , Nτ(parser, "N", "Number of timesteps for f0 correction", {"Nt"})
{
}

auto f0Args::Get() -> rl::f0Opts { return rl::f0Opts{.τ0 = τ0.Get(), .τacq = τacq.Get(), .Nτ = Nτ.Get()}; }
