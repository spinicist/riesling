#include "sense/sense.hpp"

#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "op/fft.hpp"
#include "op/nufft.hpp"
#include "op/pad.hpp"
#include "op/sense.hpp"
#include "precon.hpp"
#include "sys/threads.hpp"
#include "tensors.hpp"

namespace rl {
namespace SENSE {

Opts::Opts(args::Subparser &parser)
  : type(parser, "T", "SENSE type (auto/file.h5)", {"sense", 's'}, "auto")
  , volume(parser, "V", "SENSE calibration volume (first)", {"sense-vol"}, 0)
  , kWidth(parser, "K", "SENSE kernel width (21)", {"sense-width"}, 10)
  , res(parser, "R", "SENSE calibration res (6,6,6)", {"sense-res"}, Eigen::Array3f::Constant(6.f))
  , λ(parser, "L", "SENSE regularization (1e-6)", {"sense-lambda"}, 1.e-6f)
  , decant(parser, "D", "Direct Virtual Coil (SENSE via convolution)", {"decant"})
{
}

auto LoresChannels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis::CPtr basis) -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nS = noncart.dimension(3);
  auto const nT = noncart.dimension(4);
  if (opts.volume.Get() >= nT) { throw Log::Failure("SENSE", "Specified volume was {} data has {}", opts.volume.Get(), nT); }

  Cx4 const ncVol = noncart.chip<4>(opts.volume.Get());
  auto [traj, lores] = inTraj.downsample(ncVol, opts.res.Get(), 0, true, false);
  auto const shape1 = traj.matrixForFOV(gridOpts.fov.Get());
  auto const A = TOps::NUFFTAll(gridOpts, traj, nC, nS, 1, basis, shape1);
  auto const M = MakeKspacePre(traj, nC, nS, 1, basis);
  LSMR const lsmr{A, M, 4};

  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  Cx5 const channels = Tensorfy(lsmr.run(CollapseToConstVector(lores)), A->ishape).chip<5>(0);

  return channels;
}

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5
{
  Sz5 const shape = channels.dimensions();
  Log::Print("SENSE", "Normalizing λ {}", λ);
  Cx5 normalized(shape);
  normalized.device(Threads::TensorDevice()) =
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
 * 
 * Images need to be zero-padded to the correct oversampled grid matrix
 * 
 * The kernel width is specified on the nominal grid, i.e. will be multiplied up by the oversampling (and made odd)
 * 
 */
auto EstimateKernels(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const λ) -> Cx5
{
  if (LastN<3>(nomChan.dimensions()) != LastN<3>(nomRef.dimensions())) {
    throw Log::Failure("SENSE", "Dimensions don't match channels {} reference {}", nomChan.dimensions(), nomRef.dimensions());
  }

  Index const kW = std::floor(nomKW * osamp / 2) * 2 + 1;
  Sz3 const osshape = MulToEven(LastN<3>(nomChan.dimensions()), osamp);
  Sz5 const cshape = AddFront(osshape, nomChan.dimension(0), nomChan.dimension(1));
  Sz4 const rshape = AddFront(osshape, nomRef.dimension(0));
  
  Cx5 const channels = TOps::Pad<Cx, 5>(nomChan.dimensions(), cshape).forward(nomChan);
  Cx4 const ref = TOps::Pad<Cx, 4>(nomRef.dimensions(), rshape).forward(nomRef);

  if (cshape[2] < (2 * kW) || cshape[3] < (2 * kW) || cshape[4] < (2 * kW)) {
    throw Log::Failure("SENSE", "Matrix {} insufficient to satisfy kernel size {}", LastN<3>(cshape), kW);
  }
  Sz5 const   kshape{cshape[0], cshape[1], kW, kW, kW};
  float const scale = Norm(ref);
  Log::Print("SENSE", "Kernel shape {} scale {}", kshape, scale);
  // Set up operators
  auto D = std::make_shared<Ops::DiagScale<Cx>>(Product(kshape), std::sqrt(Product(LastN<3>(cshape)) / (float)(kW * kW * kW)));
  auto P = std::make_shared<TOps::Pad<Cx, 5>>(kshape, cshape);
  auto F = std::make_shared<TOps::FFT<5, 3>>(cshape, true);
  auto FP = std::make_shared<Ops::Multiply<Cx>>(std::make_shared<Ops::Multiply<Cx>>(F, P), D);
  auto FPinv = FP->inverse();
  auto S = std::make_shared<TOps::EstimateKernels>(ref / Cx(scale), cshape[1]);
  auto SFP = std::make_shared<Ops::Multiply<Cx>>(S, FP);
  auto PFSFP = std::make_shared<Ops::Multiply<Cx>>(FPinv, SFP);

  // Smoothness penalthy (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
  Cx3 const  sw = SobolevWeights(kW, 4).cast<Cx>();
  auto const swv = CollapseToConstVector(sw);
  auto       W = std::make_shared<Ops::DiagRep<Cx>>(swv, kshape[0] * kshape[1], 1);
  auto       L = std::make_shared<Ops::DiagScale<Cx>>(W->rows(), λ);
  auto       R = std::make_shared<Ops::Multiply<Cx>>(L, W);
  auto       A = std::make_shared<Ops::VStack<Cx>>(PFSFP, R);

  // Data
  Ops::Op<Cx>::CMap   c(channels.data(), SFP->rows());
  Ops::Op<Cx>::Vector cʹ(A->rows());
  cʹ.head(FPinv->rows()) = FPinv->forward(c) / Cx(scale);
  cʹ.tail(R->rows()).setZero();

  LSQR lsqr{A};
  lsqr.iterLimit = 16;
  auto const kʹ = lsqr.run(cʹ);

  Cx5 const kernels = Tensorfy(kʹ, kshape);
  return kernels;
}

auto KernelsToMaps(Cx5 const &kernels, Sz3 const mat, float const os) -> Cx5
{
  auto const  kshape = kernels.dimensions();
  auto const  fshape = AddFront(MulToEven(mat, os), kshape[0], kshape[1]);
  auto const  cshape = AddFront(mat, kshape[0], kshape[1]);
  float const scale = std::sqrt(Product(LastN<3>(fshape)) / (float)Product(LastN<3>(kshape)));
  Log::Print("SENSE", "Kernel Shape {} Full map shape {} Cropped map shape {} Scale {}", kshape, fshape, cshape, scale);
  TOps::Pad<Cx, 5>  P(kshape, fshape);
  TOps::FFT<5, 3>   F(fshape, false);
  TOps::Crop<Cx, 5> C(fshape, cshape);
  return C.forward(F.adjoint(P.forward(kernels))) * Cx(scale);
}

auto Choose(Opts &opts, GridOpts &gopts, Trajectory const &traj, Cx5 const &noncart) -> Cx5
{
  Cx5 kernels;
  if (noncart.dimension(0) < 2) { throw Log::Failure("SENSE", "Data is single-channel"); }
  if (opts.type.Get() == "auto") {
    Log::Print("SENSE", "Self-Calibration");
    Cx5 const c = LoresChannels(opts, gopts, traj, noncart);
    Cx4 const ref = DimDot<1>(c, c).sqrt();
    kernels = EstimateKernels(c, ref, opts.kWidth.Get(), gopts.osamp.Get(), opts.λ.Get());
  } else {
    HD5::Reader senseReader(opts.type.Get());
    kernels = senseReader.readTensor<Cx5>(HD5::Keys::Data);
  }
  return kernels;
}

} // namespace SENSE
} // namespace rl
