#include "sense.hpp"

#include "../algo/lsmr.hpp"
#include "../algo/otsu.hpp"
#include "../algo/stats.hpp"
#include "../fft.hpp"
#include "../filter.hpp"
#include "../io/hd5.hpp"
#include "../op/fft.hpp"
#include "../op/loopify.hpp"
#include "../op/mask.hpp"
#include "../op/nufft.hpp"
#include "../op/pad.hpp"
#include "../op/sense.hpp"
#include "../op/tensorscale.hpp"
#include "../precon.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

#include <flux.hpp>

namespace rl {
namespace SENSE {

std::unordered_map<std::string, SENSE::Normalization> NormMap{{"rss", SENSE::Normalization::RSS},
                                                          {"none", SENSE::Normalization::None}};

template <int ND> auto
LoresChannels(Opts<ND> const &opts, GridOpts<ND> const &gridOpts, TrajectoryN<ND> traj, Cx5 const &noncart, Basis::CPtr basis)
  -> Cx5
{
  auto const nC = noncart.dimension(0);
  // auto const nSamp = noncart.dimension(1);
  // auto const nTrace = noncart.dimension(2);
  auto const nSlice = noncart.dimension(3);
  auto const nTime = noncart.dimension(4);
  if (opts.tp >= nTime) { throw Log::Failure("SENSE", "Specified volume was {} data has {}", opts.tp, nTime); }
  Cx4 lores = traj.trim(CChipMap(noncart, opts.tp));

  traj.downsample(opts.res, true, false);
  auto const M = *flux::max(traj.matrix());
  NoncartesianTukey(M * 0.5, M, 0.f, traj.points(), lores);

  auto const nufft = TOps::MakeNUFFT<ND>(gridOpts, traj, nC, nullptr);
  Cx5        channels;
  if constexpr (ND == 2) {
    auto const A = TOps::MakeLoop<2, 3>(nufft, nSlice);
    auto const P = MakeKSpacePrecon(PreconOpts(), gridOpts, traj, nC, Sz1{nSlice});
    auto const v = CollapseToConstVector(lores);
    Log::Debug("SENSE", "A {}->{} M {}->{} Data {}", A->ishape, A->oshape, P->ishape, P->oshape, lores.dimensions());
    LSMR const lsmr{A, P, nullptr, {opts.its}};
    channels = AsTensorMap(lsmr.run(v), A->ishape);
  } else {
    if (nSlice > 1) { throw(Log::Failure("SENSE", "Not supported right now")); }
    auto const P = MakeKSpacePrecon(PreconOpts(), gridOpts, traj, nC, Sz0{});
    LSMR const lsmr{nufft, P, nullptr, {opts.its}};
    channels = AsTensorMap(lsmr.run(CollapseToConstVector(lores)), nufft->ishape);
  }
  return channels;
}

template auto
LoresChannels(Opts<2> const &opts, GridOpts<2> const &gridOpts, TrajectoryN<2> traj, Cx5 const &noncart, Basis::CPtr basis)
  -> Cx5;
template auto
LoresChannels(Opts<3> const &opts, GridOpts<3> const &gridOpts, TrajectoryN<3> traj, Cx5 const &noncart, Basis::CPtr basis)
  -> Cx5;

void Normalize(Cx5 &maps)
{
  Sz5 const shape = maps.dimensions();
  Sz4 const shape4 = FirstN<4>(shape);
  Log::Print("SENSE", "Normalizing to RSS");
  Cx4 rss(shape4);
  rss.device(Threads::TensorDevice()) = DimDot<4>(maps, maps).sqrt();
  maps.device(Threads::TensorDevice()) = maps / rss.reshape(AddBack(shape4, 1)).broadcast(Sz5{1, 1, 1, 1, shape[4]});
}

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5
{
  Sz5 const shape = channels.dimensions();
  Log::Print("SENSE", "Normalizing λ {}", λ);
  Cx5 normalized(shape);
  normalized.device(Threads::TensorDevice()) =
    channels / (ref + ref.constant(λ)).reshape(AddBack(FirstN<4>(shape), 1)).broadcast(Sz5{1, 1, 1, 1, shape[4]});
  return normalized;
}

template <int ND> auto SobolevWeights(Index const kW, Index const l) -> ReN<ND>
{
  Sz<ND> shape;
  std::fill_n(shape.begin(), ND, kW);
  ReN<ND> W(shape);
  for (Index ik = 0; ik < kW; ik++) {
    float const kk = (ik - (kW / 2));
    for (Index ij = 0; ij < kW; ij++) {
      float const kj = (ij - (kW / 2));
      if constexpr (ND == 2) {
        float const k2 = (kj * kj + kk * kk);
        W(ij, ik) = std::pow(1.f + k2, l / 2);
      } else {
        for (Index ii = 0; ii < kW; ii++) {
          float const ki = (ii - (kW / 2));
          float const k2 = (ki * ki + kj * kj + kk * kk);
          W(ii, ij, ik) = std::pow(1.f + k2, l / 2);
        }
      }
    }
  }
  return W;
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
template <int ND> auto EstimateKernels(
  Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ, Normalization const renorm)
  -> Cx5
{
  if (FirstN<3>(nomChan.dimensions()) != FirstN<3>(nomRef.dimensions())) {
    throw Log::Failure("SENSE", "Dimensions don't match channels {} reference {}", nomChan.dimensions(), nomRef.dimensions());
  }
  if (nomChan.dimension(3) != nomRef.dimension(3)) {
    throw Log::Failure("SENSE", "Basis dimension doesn't match channels {} reference {}", nomChan.dimension(3),
                       nomRef.dimension(3));
  }
  Index const nB = nomChan.dimension(3);
  Index const nC = nomChan.dimension(4);

  Index const kW = std::floor(nomKW * osamp / 2) * 2 + 1;
  Sz3         osshape;
  if constexpr (ND == 2) {
    osshape = AddBack(MulToEven(FirstN<2>(nomChan.dimensions()), osamp), nomChan.dimension(2));
  } else {
    osshape = MulToEven(FirstN<3>(nomChan.dimensions()), osamp);
  }
  Sz5 const cshape = AddBack(osshape, nB, nC);
  Sz4 const rshape = AddBack(osshape, nB);

  Cx5 const channels = TOps::Pad<5>(nomChan.dimensions(), cshape).forward(nomChan) / Cx(Norm<true>(nomChan));
  Cx4 const ref = TOps::Pad<4>(nomRef.dimensions(), rshape).forward(nomRef) / Cx(Norm<true>(nomRef));

  Sz5 kshape;
  if constexpr (ND == 2) {
    if (cshape[0] < (2 * kW) || cshape[1] < (2 * kW)) {
      throw Log::Failure("SENSE", "Matrix {} insufficient to satisfy kernel size {}", FirstN<2>(cshape), kW);
    }
    kshape = {kW, kW, cshape[2], cshape[3], cshape[4]};
  } else {
    if (cshape[0] < (2 * kW) || cshape[1] < (2 * kW) || cshape[2] < (2 * kW)) {
      throw Log::Failure("SENSE", "Matrix {} insufficient to satisfy kernel size {}", FirstN<3>(cshape), kW);
    }
    kshape = {kW, kW, kW, cshape[3], cshape[4]};
  }

  Log::Print("SENSE", "Kernel shape {}", kshape);
  // Set up operators
  auto D = Ops::DiagScale::Make(Product(kshape), std::sqrt(Product(FirstN<ND>(cshape)) / std::pow(kW, ND)));
  auto P = TOps::Pad<5>::Make(kshape, cshape);
  auto F = TOps::FFT<5, ND>::Make(cshape, true);
  auto FP = Ops::Mul(Ops::Mul(F, P), D);
  auto S = TOps::TensorScale<5, 0, 1>::Make(cshape, ref);
  auto SFP = Ops::Mul(S, FP);

  // Weights (mask)
  auto const           r = CollapseToArray(ref);
  Eigen::ArrayXf const ra = r.abs();
  Eigen::ArrayXf const rm = OtsuMasked(ra);
  auto                 M = Ops::Mask::Make(rm, nB * nC);
  auto                 MSFP = Ops::Mul(M, SFP);

  Cx5 kernels;
  if (λ > 0.f) {
    // Smoothness penalthy (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
    CxN<ND> const sw = SobolevWeights<ND>(kW, l).template cast<Cx>();
    auto const    swv = CollapseToConstVector(sw);
    auto          W = std::make_shared<Ops::DiagRep>(swv, 1, Product(LastN<5 - ND>(kshape)));
    auto          L = std::make_shared<Ops::DiagScale>(W->rows(), λ);

    // Combine
    auto A = Ops::VStack::Make({MSFP, Ops::Mul(L, W)});

    // Preconditioner
    Ops::Op::Vector p(A->rows());
    p.setConstant(1.f);
    p = (A->forward(A->adjoint(p)).array().abs() + 1.e-3f).inverse();
    auto R = std::make_shared<Ops::DiagRep>(p, 1, 1);

    // Data
    Ops::Op::CMap   c(channels.data(), SFP->rows());
    Ops::Op::Vector cʹ(A->rows());
    cʹ.head(MSFP->rows()) = M->forward(c);
    cʹ.tail(L->rows()).setZero();

    LSMR       solve{A, R, nullptr, LSMR::Opts{.imax = 256, .aTol = 1e-4f}};
    auto const kʹ = solve.run(cʹ);

    kernels = AsTensorMap(kʹ, kshape);
  } else {
    // Preconditioner
    Ops::Op::Vector p(MSFP->rows());
    p.setConstant(1.f);
    p = (MSFP->forward(MSFP->adjoint(p)).array().abs() + 1.e-3f).inverse();
    auto            R = std::make_shared<Ops::DiagRep>(p, 1, 1);
    Ops::Op::CMap   c(channels.data(), SFP->rows());
    Ops::Op::Vector cʹ = M->forward(c);
    LSMR            solve{MSFP, R, nullptr, LSMR::Opts{.imax = 256, .aTol = 1e-4f}};
    auto const      k = solve.run(cʹ);
    kernels = AsTensorMap(k, kshape);
  }

  if (renorm == Normalization::RSS) {
    auto const mshape = MulToEven(FirstN<ND>(kshape), 2.f);
    auto const maps = KernelsToMaps(kernels, mshape, 1.f, renorm);
    kernels = MapsToKernels(maps, mshape, 1.f);
  }

  return kernels;
}

template auto
EstimateKernels<2>(Cx5 const &, Cx4 const &, Index const, float const, float const, float const, Normalization const) -> Cx5;
template auto
EstimateKernels<3>(Cx5 const &, Cx4 const &, Index const, float const, float const, float const, Normalization const) -> Cx5;

/*
 * For 2D, assume multislice so only inflate the first two dimensions
 */
template <int ND> auto KernelsToMaps(Cx5 const &kernels, Sz<ND> const mat, float const os, Normalization const renorm) -> Cx5
{
  auto const  kshape = kernels.dimensions();
  auto const  fshape = Concatenate(MulToEven(mat, os), LastN<5 - ND>(kshape));
  auto const  mshape = Concatenate(mat, LastN<5 - ND>(kshape));
  float const scale = std::sqrt(Product(FirstN<ND>(fshape)) / (float)Product(FirstN<ND>(kshape)));
  Log::Debug("SENSE", "Kernels {} Oversampled maps {} Maps {} Scale {}", kshape, fshape, mshape, scale);
  TOps::Pad<5>     P(kshape, fshape);
  TOps::FFT<5, ND> F(fshape, false);
  TOps::Pad<5>     C(mshape, fshape);
  Cx5              maps = C.adjoint(F.adjoint(P.forward(kernels))) * Cx(scale);
  if (renorm == Normalization::RSS) { Normalize(maps); }
  return maps;
}

template auto KernelsToMaps(Cx5 const &, Sz1 const, float const, Normalization const) -> Cx5;
template auto KernelsToMaps(Cx5 const &, Sz2 const, float const, Normalization const) -> Cx5;
template auto KernelsToMaps(Cx5 const &, Sz3 const, float const, Normalization const) -> Cx5;

template <int ND> auto MapsToKernels(Cx5 const &maps, Sz<ND> const kmat, float const os) -> Cx5
{
  auto const  mshape = maps.dimensions();
  auto const  fshape = Concatenate(MulToEven(FirstN<ND>(mshape), os), LastN<5 - ND>(mshape));
  auto const  kshape = Concatenate(kmat, LastN<5 - ND>(mshape));
  float const scale = std::sqrt(Product(FirstN<ND>(fshape)) / (float)Product(FirstN<ND>(kshape)));
  Log::Debug("SENSE", "Maps {} Oversampled maps {} Kernels {} Scale {}", mshape, fshape, kshape, scale);
  TOps::Pad<5>    P(mshape, fshape);
  TOps::FFT<5, 3> F(fshape, true);
  TOps::Pad<5>    C(kshape, fshape);
  return C.adjoint(F.adjoint(P.forward(maps))) / Cx(scale);
}
template auto MapsToKernels(Cx5 const &, Sz2 const, float const) -> Cx5;
template auto MapsToKernels(Cx5 const &, Sz3 const, float const) -> Cx5;

template <int ND> auto Choose(Opts<ND> const &opts, GridOpts<ND> const &gopts, TrajectoryN<ND> const &traj, Cx5 const &noncart)
  -> Cx5
{
  Cx5 kernels;
  if (noncart.dimension(0) < 2) { throw Log::Failure("SENSE", "Data is single-channel"); }
  if (opts.type == "auto") {
    Log::Print("SENSE", "Self-Calibration");
    Cx5 const c = LoresChannels<ND>(opts, gopts, traj, noncart);
    Cx4 const ref = DimDot<4>(c, c).sqrt();
    kernels = EstimateKernels<ND>(c, ref, opts.kWidth, gopts.osamp, opts.l, opts.λ, opts.renorm);
  } else {
    HD5::Reader senseReader(opts.type);
    kernels = senseReader.readTensor<Cx5>(HD5::Keys::Data);
    if (kernels.dimension(4) != noncart.dimension(0)) {
      throw(Log::Failure("SENSE", "Kernel channels {} did not match data {}", kernels.dimension(4), noncart.dimension(0)));
    }
  }
  return kernels;
}

template auto Choose(Opts<2> const &opts, GridOpts<2> const &gopts, TrajectoryN<2> const &traj, Cx5 const &noncart) -> Cx5;
template auto Choose(Opts<3> const &opts, GridOpts<3> const &gopts, TrajectoryN<3> const &traj, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
