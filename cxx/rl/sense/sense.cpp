#include "sense.hpp"

#include "../algo/lsmr.hpp"
#include "../algo/otsu.hpp"
#include "../algo/stats.hpp"
#include "../fft.hpp"
#include "../filter.hpp"
#include "../io/hd5.hpp"
#include "../op/fft.hpp"
#include "../op/nufft.hpp"
#include "../op/pad.hpp"
#include "../op/sense.hpp"
#include "../op/tensorscale.hpp"
#include "../precon.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl {
namespace SENSE {

auto LoresChannels(
  Opts const &opts, GridOpts<3> const &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis::CPtr basis) -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nS = noncart.dimension(3);
  auto const nT = noncart.dimension(4);
  if (opts.tp >= nT) { throw Log::Failure("SENSE", "Specified volume was {} data has {}", opts.tp, nT); }

  Cx4 const ncVol = noncart.chip<4>(opts.tp);
  auto [traj, lores] = inTraj.downsample(ncVol, opts.res, 0, true, false);
  auto       sgOpts = gridOpts;
  auto const A = TOps::NUFFTAll(sgOpts, traj, nC, nS, 1, nullptr);
  auto const M = MakeKSpacePrecon(PreconOpts(), sgOpts, traj, nC, nS, nT);
  LSMR const lsmr{A, M, nullptr, {4}};

  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  Cx5 const channels = AsTensorMap(lsmr.run(CollapseToConstVector(lores)), A->ishape).chip<5>(0);

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

auto SobolevWeights(Sz3 const shape, Index const l) -> Re3
{
  Re3 W(shape);
  for (Index ik = 0; ik < shape[2]; ik++) {
    float const kk = (ik - (shape[2] / 2));
    for (Index ij = 0; ij < shape[1]; ij++) {
      float const kj = (ij - (shape[1] / 2));
      for (Index ii = 0; ii < shape[0]; ii++) {
        float const ki = (ii - (shape[0] / 2));
        float const k2 = (ki * ki + kj * kj + kk * kk);
        W(ii, ij, ik) = std::pow(1.f + k2, l / 2);
      }
    }
  }
  return W;
}

/* We want to solve:
 * c = R s
 *
 * Where c is the channel images, s is SENSE maps, R is multiply by reference image
 *
 * We need to add the regularizer from nlinv / ENLIVE. This is a Sobolev weights in k-space
 * P is padding, F is FT
 * Hence solve the modified system
 *
 * c' = [ c ] = [ R    ]  s = A s
 *      [ 0 ]   [ λW F ]
 * A = [ R    ]
 *     [ λW F ]
 *
 * Needs a pre-conditioner. Use a right preconditioner N = [ I + R ]
 *
 * Leaving this here for posterity, but it's easier to estimate the kernels directly due to oversampling
 *
 */
auto EstimateMaps(Cx5 const &ichan, Cx4 const &iref, float const osamp, float const l, float const λ) -> Cx5
{
  if (FirstN<3>(ichan.dimensions()) != FirstN<3>(iref.dimensions())) {
    throw Log::Failure("SENSE", "Dimensions don't match channels {} reference {}", ichan.dimensions(), iref.dimensions());
  }

  Sz3 const osshape = MulToEven(FirstN<3>(ichan.dimensions()), osamp);
  Sz5 const cshape = AddBack(osshape, ichan.dimension(3), ichan.dimension(4));
  Sz4 const rshape = AddBack(osshape, iref.dimension(3));

  // Need to swap channel and basis dimensions to make TensorScale work
  Cx5 const chan = TOps::Pad<Cx, 5>(ichan.dimensions(), cshape).forward(ichan).shuffle(Sz5{0, 1, 2, 4, 3});
  Cx4 const ref = TOps::Pad<Cx, 4>(iref.dimensions(), rshape).forward(iref);

  auto const mapshape = chan.dimensions();
  Log::Print("SENSE", "Map shape {}", mapshape);
  // Need various scaling factors
  float const nref = Norm<true>(ref);
  float const median = Percentiles(OtsuMask(CollapseToArray(ref).abs()), {0.5}).front();
  Cx4 const   w = (ref / Cx(median) + Cx(1.f)).log();
  // Weighted Least Squares
  auto R = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(mapshape, ref / Cx(nref));
  auto W = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(mapshape, w);
  auto Wr = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(mapshape, w.sqrt());
  auto WrR = Ops::Mul<Cx>(Wr, R);
  // Smoothness regularizer (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
  auto F = std::make_shared<TOps::FFT<5, 3>>(mapshape, Sz3{0, 1, 2});
  auto K = std::make_shared<TOps::TensorScale<Cx, 5, 0, 2>>(mapshape, SobolevWeights(FirstN<3>(mapshape), l).cast<Cx>());
  auto L = std::make_shared<Ops::DiagScale<Cx>>(K->rows(), λ);
  auto LKF = Ops::Mul<Cx>(L, Ops::Mul<Cx>(K, F));
  // Combine operators
  auto A = std::make_shared<Ops::VStack<Cx>>(R, LKF);
  // Data
  Ops::Op<Cx>::CMap   c(chan.data(), chan.size());
  Ops::Op<Cx>::Vector cʹ(A->rows());
  cʹ.head(c.size()) = c / Cx(nref);
  cʹ.tail(LKF->rows()).setZero();
  // Preconditioner
  Ops::Op<Cx>::Vector m(A->rows());
  m.setConstant(1.f);
  m = (A->forward(A->adjoint(m)).array().abs() + 1.e-3f).inverse();
  auto Minv = std::make_shared<Ops::DiagRep<Cx>>(m, 1, 1);

  LSMR       solve{A, Minv, nullptr, LSMR::Opts{.imax = 256, .aTol = 1e-6f}};
  auto const s = solve.run(cʹ);
  Cx5        maps = AsTensorMap(s, mapshape);
  return maps.shuffle(Sz5{0, 1, 2, 4, 3});
}

/* We want to solve:
 * c = R F P k
 *
 * Where c is the channel images, k is the SENSE kernels
 * P is padding, F is FT, R is multiply by the reference image
 * This is a circular convolution done via FFT
 *
 * We need to add:
 * 1 - the regularizer from nlinv / ENLIVE. This is Sobolev Norm / weighted k-space
 * 2 - a mask/weights (M) to exclude the background region. Define this as 1s/0s then
 *     don't need to worry about the sqrt() in the system matrix.
 * Hence solve the modified system
 *
 * c' = [ c ] = [ M R F P ]  k
 *      [ 0 ]   [      λW ]
 * A = [ M R F P ]
 *     [      λW ]
 *
 * Images need to be zero-padded to the correct oversampled grid matrix
 *
 * The kernel width is specified on the nominal grid, i.e. will be multiplied up by the oversampling (and made odd)
 *
 */
auto EstimateKernels(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5
{
  if (FirstN<3>(nomChan.dimensions()) != FirstN<3>(nomRef.dimensions())) {
    throw Log::Failure("SENSE", "Dimensions don't match channels {} reference {}", nomChan.dimensions(), nomRef.dimensions());
  }
  Cx5 const schan = nomChan.shuffle(Sz5{0, 1, 2, 4, 3});

  Index const kW = std::floor(nomKW * osamp / 2) * 2 + 1;
  Sz3 const   osshape = MulToEven(FirstN<3>(schan.dimensions()), osamp);
  Sz5 const   cshape = AddBack(osshape, schan.dimension(3), schan.dimension(4));
  Sz4 const   rshape = AddBack(osshape, nomRef.dimension(3));

  float const scale = Norm<true>(nomRef);
  Cx5 const   channels = TOps::Pad<Cx, 5>(schan.dimensions(), cshape).forward(schan) / Cx(scale);
  Cx4 const   ref = TOps::Pad<Cx, 4>(nomRef.dimensions(), rshape).forward(nomRef) / Cx(scale);

  if (cshape[0] < (2 * kW) || cshape[1] < (2 * kW) || cshape[2] < (2 * kW)) {
    throw Log::Failure("SENSE", "Matrix {} insufficient to satisfy kernel size {}", FirstN<3>(cshape), kW);
  }
  Sz5 const kshape{kW, kW, kW, cshape[3], cshape[4]};
  Log::Print("SENSE", "Kernel shape {} scale {}", kshape, scale);
  // Set up operators
  auto D = std::make_shared<Ops::DiagScale<Cx>>(Product(kshape), std::sqrt(Product(FirstN<3>(cshape)) / (float)(kW * kW * kW)));
  auto P = std::make_shared<TOps::Pad<Cx, 5>>(kshape, cshape);
  auto F = std::make_shared<TOps::FFT<5, 3>>(cshape, true);
  auto FP = Ops::Mul<Cx>(Ops::Mul<Cx>(F, P), D);
  auto S = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(cshape, ref);
  auto SFP = Ops::Mul<Cx>(S, FP);

  // Weights (mask)
  auto const            r = CollapseToArray(ref);
  Eigen::ArrayXf const  ra = r.abs();
  Eigen::ArrayXcf const rm = OtsuMasked(ra).cast<Cx>();
  auto const            om = AsConstTensorMap(rm, rshape);
  auto                  M = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(cshape, om);
  auto                  MSFP = Ops::Mul<Cx>(M, SFP);

  Cx5 kernels;
  if (λ > 0.f) {
    // Smoothness penalthy (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
    Cx3 const  sw = SobolevWeights(kW, l).cast<Cx>();
    auto const swv = CollapseToConstVector(sw);
    auto       W = std::make_shared<Ops::DiagRep<Cx>>(swv, 1, kshape[3] * kshape[4]);
    auto       L = std::make_shared<Ops::DiagScale<Cx>>(W->rows(), λ);

    // Combine
    auto A = std::make_shared<Ops::VStack<Cx>>(MSFP, Ops::Mul<Cx>(L, W));

    // Preconditioner
    Ops::Op<Cx>::Vector m(A->rows());
    m.setConstant(1.f);
    m = (A->forward(A->adjoint(m)).array().abs() + 1.e-3f).inverse();
    auto Minv = std::make_shared<Ops::DiagRep<Cx>>(m, 1, 1);

    // Data
    Ops::Op<Cx>::CMap   c(channels.data(), SFP->rows());
    Ops::Op<Cx>::Vector cʹ(A->rows());
    cʹ.head(MSFP->rows()) = M->forward(c);
    cʹ.tail(L->rows()).setZero();

    LSMR       solve{A, Minv, nullptr, LSMR::Opts{.imax = 256, .aTol = 1e-4f}};
    auto const kʹ = solve.run(cʹ);

    kernels = AsTensorMap(kʹ, kshape);
  } else {
    // Preconditioner
    Ops::Op<Cx>::Vector m(MSFP->rows());
    m.setConstant(1.f);
    m = (MSFP->forward(MSFP->adjoint(m)).array().abs() + 1.e-3f).inverse();
    auto                Minv = std::make_shared<Ops::DiagRep<Cx>>(m, 1, 1);
    Ops::Op<Cx>::CMap   c(channels.data(), MSFP->rows());
    Ops::Op<Cx>::Vector cʹ = M->forward(c);
    LSMR                solve{MSFP, Minv, nullptr, LSMR::Opts{.imax = 256, .aTol = 1e-4f}};
    auto const          k = solve.run(cʹ);
    kernels = AsTensorMap(k, kshape);
  }
  return kernels.shuffle(Sz5{0, 1, 2, 4, 3});
}

auto KernelsToMaps(Cx5 const &kernels, Sz3 const mat, float const os) -> Cx5
{
  auto const  kshape = kernels.dimensions();
  auto const  fshape = AddBack(MulToEven(mat, os), kshape[3], kshape[4]);
  auto const  cshape = AddBack(mat, kshape[3], kshape[4]);
  float const scale = std::sqrt(Product(FirstN<3>(fshape)) / (float)Product(FirstN<3>(kshape)));
  Log::Print("SENSE", "Kernels {} Full maps {} Cropped maps {} Scale {}", kshape, fshape, cshape, scale);
  TOps::Pad<Cx, 5> P(kshape, fshape);
  TOps::FFT<5, 3>  F(fshape, false);
  TOps::Pad<Cx, 5> C(cshape, fshape);
  return C.adjoint(F.adjoint(P.forward(kernels))) * Cx(scale);
}

auto MapsToKernels(Cx5 const &maps, Index const nomKW, float const os) -> Cx5
{
  Index const kW = std::floor(nomKW * os / 2) * 2 + 1;
  auto const  mshape = maps.dimensions();
  auto const  oshape = AddBack(MulToEven(FirstN<3>(mshape), os), mshape[3], mshape[4]);
  auto const  kshape = Sz5{kW, kW, kW, mshape[3], mshape[4]};
  float const scale = std::sqrt(Product(FirstN<3>(oshape)) / (float)Product(FirstN<3>(mshape)));
  Log::Print("SENSE", "Map Shape {} Oversampled map shape {} Kernel shape {} Scale {}", mshape, oshape, kshape, scale);
  TOps::Pad<Cx, 5> P(mshape, oshape);
  TOps::FFT<5, 3>  F(oshape, true);
  TOps::Pad<Cx, 5> C(kshape, oshape);
  return C.adjoint(F.adjoint(P.forward(maps))) * Cx(scale);
}

auto Choose(Opts const &opts, GridOpts<3> const &gopts, Trajectory const &traj, Cx5 const &noncart) -> Cx5
{
  Cx5 kernels;
  if (noncart.dimension(0) < 2) { throw Log::Failure("SENSE", "Data is single-channel"); }
  if (opts.type == "auto") {
    Log::Print("SENSE", "Self-Calibration");
    Cx5 const c = LoresChannels(opts, gopts, traj, noncart);
    Cx4 const ref = DimDot<3>(c, c).sqrt();
    kernels = EstimateKernels(c, ref, opts.kWidth, gopts.osamp, opts.l, opts.λ);
  } else {
    HD5::Reader senseReader(opts.type);
    kernels = senseReader.readTensor<Cx5>(HD5::Keys::Data);
  }
  return kernels;
}

} // namespace SENSE
} // namespace rl
