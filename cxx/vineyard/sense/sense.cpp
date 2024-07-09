#include "sense/sense.hpp"

#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "op/fft.hpp"
#include "op/pad.hpp"
#include "op/recon.hpp"
#include "op/sense.hpp"
#include "precon.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl {
namespace SENSE {

Opts::Opts(args::Subparser &parser)
  : type(parser, "T", "SENSE type (auto/EstimateKernels/file.h5)", {"sense", 's'}, "auto")
  , volume(parser, "V", "SENSE calibration volume (first)", {"sense-vol"}, 0)
  , kWidth(parser, "K", "SENSE kernel width (21)", {"sense-width"}, 21)
  , res(parser, "R", "SENSE calibration res (6,6,6)", {"sense-res"}, Eigen::Array3f::Constant(6.f))
  , fov(parser, "SENSE-FOV", "SENSE FOV (default header FOV)", {"sense-fov"}, Eigen::Array3f::Zero())
  , λ(parser, "L", "SENSE regularization (1e-3)", {"sense-lambda"}, 1.e-3f)
{
}

auto LoresChannels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis<Cx> const &basis) -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nS = noncart.dimension(3);
  auto const nV = noncart.dimension(4);
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }

  Cx4 const ncVol = noncart.chip<4>(opts.volume.Get());
  auto [traj, lores] = inTraj.downsample(ncVol, opts.res.Get(), 0, true, false);
  auto const shape1 = traj.matrix(gridOpts.osamp.Get());
  auto const A = Recon::Channels(false, gridOpts, traj, nC, nS, basis, shape1);
  auto const M = make_kspace_pre(traj, nC, basis, gridOpts.vcc);
  LSMR const lsmr{A, M, 4};

  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  Cx5 const channels(Tensorfy(lsmr.run(lores.data()), A->ishape));

  return channels;
}

auto LoresKernels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis<Cx> const &basis) -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nV = noncart.dimension(4);
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }

  Sz3 kSz;
  kSz.fill(opts.kWidth.Get());
  Cx4 const ncVol = noncart.chip<4>(opts.volume.Get());
  auto const [traj, lores] = inTraj.downsample(ncVol, kSz, 0, true, true);
  auto const A = TOps::Grid<Cx, 3>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis);
  auto const M = make_kspace_pre(traj, nC, basis, false);
  LSMR const lsmr{A, M, 4};
  Cx5 const  channels(Tensorfy(lsmr.run(lores.data()), A->ishape));
  return channels;
}

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5
{
  Sz5 const shape = channels.dimensions();
  Log::Print("Normalizing SENSE. Dimensions {} λ {}", shape, λ);
  Cx5 normalized(shape);
  normalized.device(Threads::GlobalDevice()) =
    channels / (ref + ref.constant(λ)).reshape(AddFront(LastN<4>(shape), 1)).broadcast(Sz5{shape[0], 1, 1, 1, 1});
  return normalized;
}

auto SobolevWeights(Index const kW, Index const l) -> Re3
{
  Re3 W(kW, kW, kW);
  for (Index ik = 0; ik < kW; ik++) {
    float const kk = (ik - (kW / 2));
    for (Index ij = 0; ij < kW; ij++) {
      float const kj = (ij - (kW / 2));
      for (Index ii = 0; ii < kW; ii++) {
        float const ki = (ii - (kW / 2));
        float const k2 = (ki * ki + kj * kj + kk * kk);
        W(ii, ij, ik) = std::pow(1.f + k2, l / 2);
      }
    }
  }
  return W;
}

/* We want to solve:
 * Pt Ft c = Pt Ft S F P k
 *
 * Where c is the channel images, k is the SENSE kernels
 * P is padding, F is FT, S is SENSE (but multiply by the reference image)
 * This is a circular convolution done via FFT
 *
 * We need to add the regularizer from nlinv / ENLIVE. This is a Sobolev weights in k-space
 * Hence solve the modified system
 *
 * c' = [ Pt Ft c ] = [ Pt Ft S F P ]  k
 *      [       0 ]   [          λW ]
 * A = [ Pt Ft S F P ]
 *     [          λW ]
 */
auto EstimateKernels(Cx5 const &channels, Cx4 const &ref, Index const kW, float const λ) -> Cx5
{
  Sz5 const cshape = channels.dimensions();
  if (LastN<4>(cshape) != ref.dimensions()) {
    Log::Fail("SENSE dimensions don't match channels {} reference {}", cshape, ref.dimensions());
  }
  if (cshape[2] < (2 * kW) || cshape[3] < (2 * kW) || cshape[4] < (2 * kW)) {
    Log::Fail("SENSE matrix {} insufficient to satisfy kernel size {}", LastN<3>(cshape), kW);
  }
  Sz5 const kshape{cshape[0], cshape[1], kW, kW, kW};

  // Set up operators
  auto D = std::make_shared<Ops::DiagScale<Cx>>(Product(kshape), std::sqrt(Product(LastN<3>(cshape)) / (float)(kW * kW * kW)));
  auto P = std::make_shared<TOps::Pad<Cx, 5>>(kshape, cshape);
  auto F = std::make_shared<TOps::FFT<5, 3>>(cshape, true);
  auto FP = std::make_shared<Ops::Multiply<Cx>>(std::make_shared<Ops::Multiply<Cx>>(F, P), D);
  auto FPinv = FP->inverse();
  auto S = std::make_shared<TOps::EstimateKernels>(ref, cshape[0]);
  auto SFP = std::make_shared<Ops::Multiply<Cx>>(S, FP);
  auto PFSFP = std::make_shared<Ops::Multiply<Cx>>(FPinv, SFP);

  // Smoothness penalthy (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
  Cx3 const  sw = SobolevWeights(kW, 4).cast<Cx>();
  auto const swv = CollapseToArray(sw);
  auto       W = std::make_shared<Ops::DiagRep<Cx>>(kshape[0] * kshape[1], swv);
  auto       L = std::make_shared<Ops::DiagScale<Cx>>(W->rows(), λ);
  auto       R = std::make_shared<Ops::Multiply<Cx>>(L, W);
  auto       A = std::make_shared<Ops::VStack<Cx>>(PFSFP, R);

  // Data
  Ops::Op<Cx>::CMap c(channels.data(), SFP->rows());
  auto const        ck = FPinv->forward(c);

  Ops::Op<Cx>::Vector cʹ(A->rows());
  cʹ.head(FPinv->rows()) = ck;
  cʹ.tail(R->rows()).setZero();

  LSQR lsqr{A};
  lsqr.iterLimit = 16;
  auto const kʹ = lsqr.run(cʹ.data(), 0.f);

  Cx5 const kernels = Tensorfy(kʹ, kshape);
  return kernels;
}

auto KernelsToMaps(Cx5 const &kernels, Sz3 const fmat, Sz3 const cmat) -> Cx5
{
  auto const        kshape = kernels.dimensions();
  auto const        fshape = AddFront(fmat, kshape[0], kshape[1]);
  auto const        cshape = AddFront(cmat, kshape[0], kshape[1]);
  TOps::Pad<Cx, 5>  P(kshape, fshape);
  TOps::FFT<5, 3>   F(fshape, false);
  TOps::Crop<Cx, 5> C(fshape, cshape);
  return C.forward(F.adjoint(P.forward(kernels))) * Cx(std::sqrt(Product(LastN<3>(fshape)) / (float)Product(LastN<3>(kshape))));
}

auto Choose(Opts &opts, GridOpts &gopts, Trajectory const &traj, Cx5 const &noncart) -> Cx5
{
  Cx5 kernels;
  if (opts.type.Get() == "auto") {
    Log::Print("SENSE Self-Calibration");
    Cx5 const c = LoresChannels(opts, gopts, traj, noncart);
    Cx4 const ref = ConjugateSum(c, c);
    kernels = EstimateKernels(c, ref, opts.kWidth.Get(), opts.λ.Get());
  } else {
    HD5::Reader senseReader(opts.type.Get());
    kernels = senseReader.readTensor<Cx5>(HD5::Keys::Data);
  }
  return SENSE::KernelsToMaps(kernels, traj.matrix(gopts.osamp.Get()), traj.matrixForFOV(opts.fov.Get()));
}

} // namespace SENSE
} // namespace rl
