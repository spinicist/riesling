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

namespace rl {
namespace SENSE {

template <int ND> auto
LoresChannels(Opts<ND> const &opts, GridOpts<ND> const &gridOpts, TrajectoryN<ND> traj, Cx5 const &noncart, Basis::CPtr basis)
  -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nSamp = noncart.dimension(1);
  auto const nTrace = noncart.dimension(2);
  auto const nSlice = noncart.dimension(3);
  auto const nTime = noncart.dimension(4);
  if (opts.tp >= nTime) { throw Log::Failure("SENSE", "Specified volume was {} data has {}", opts.tp, nTime); }

  Cx4 const ncVol = noncart.chip<4>(opts.tp);
  traj.downsample(opts.res, true, false);
  auto       lores = traj.trim(ncVol);
  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);

  auto const nufft = TOps::NUFFT<ND>::Make(gridOpts, traj, nC, nullptr);
  Cx5        channels;
  if constexpr (ND == 2) {
    auto const A = TOps::MakeLoop<2, 3>(nufft, nSlice);
    auto const M = MakeKSpacePrecon(PreconOpts(), gridOpts, traj, nC, Sz1{nSlice});
    auto const lores =
      (nTime == 1) ? traj.trim(Cx4CMap(noncart.data(), nC, nSamp, nTrace, nSlice)) : traj.trim(Cx4(noncart.chip<4>(opts.tp)));
    auto const v = CollapseToConstVector(lores);
    Log::Debug("SENSE", "A {}->{} M {}->{} Data {}", A->ishape, A->oshape, M->ishape, M->oshape, lores.dimensions());
    LSMR const lsmr{A, M, nullptr, {4}};
    channels = AsTensorMap(lsmr.run(v), A->ishape);
  } else {
    if (nSlice > 1) { throw(Log::Failure("SENSE", "Not supported right now")); }
    auto const M = MakeKSpacePrecon(PreconOpts(), gridOpts, traj, nC, Sz0{});
    auto const lores = (nTime == 1) ? traj.trim(Cx3CMap(noncart.data(), nC, nSamp, nTrace))
                                    : traj.trim(Cx3(noncart.chip<4>(opts.tp).template chip<3>(0)));
    LSMR const lsmr{nufft, M, nullptr, {4}};
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

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5
{
  Sz5 const shape = channels.dimensions();
  Log::Print("SENSE", "Normalizing λ {}", λ);
  Cx5 normalized(shape);
  normalized.device(Threads::TensorDevice()) =
    channels / (ref + ref.constant(λ)).reshape(AddFront(LastN<4>(shape), 1)).broadcast(Sz5{shape[0], 1, 1, 1, 1});
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
template <int ND>
auto EstimateKernels(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5
{
  if (FirstN<3>(nomChan.dimensions()) != FirstN<3>(nomRef.dimensions())) {
    throw Log::Failure("SENSE", "Dimensions don't match channels {} reference {}", nomChan.dimensions(), nomRef.dimensions());
  }
  if (nomChan.dimension(4) != nomRef.dimension(3)) {
    throw Log::Failure("SENSE", "Basis dimension doesn't match channels {} reference {}", nomChan.dimension(4),
                       nomRef.dimension(4));
  }
  Index const nC = nomChan.dimension(3);
  Index const nB = nomChan.dimension(4);

  // Dims are i,j,k,c,b need them to be i,j,k,b,c
  Cx5 const schan = nomChan.shuffle(Sz5{0, 1, 2, 4, 3});

  Index const kW = std::floor(nomKW * osamp / 2) * 2 + 1;
  Sz3         osshape;
  if constexpr (ND == 2) {
    osshape = AddBack(MulToEven(FirstN<2>(schan.dimensions()), osamp), schan.dimension(2));
  } else {
    osshape = MulToEven(FirstN<3>(schan.dimensions()), osamp);
  }
  Sz5 const cshape = AddBack(osshape, nB, nC);
  Sz4 const rshape = AddBack(osshape, nB);

  Cx5 const channels = TOps::Pad<Cx, 5>(schan.dimensions(), cshape).forward(schan) / Cx(Norm<true>(schan));
  Cx4 const ref = TOps::Pad<Cx, 4>(nomRef.dimensions(), rshape).forward(nomRef) / Cx(Norm<true>(nomRef));

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
  auto D = std::make_shared<Ops::DiagScale<Cx>>(Product(kshape), std::sqrt(Product(FirstN<ND>(cshape)) / std::pow(kW, ND)));
  auto P = std::make_shared<TOps::Pad<Cx, 5>>(kshape, cshape);
  auto F = std::make_shared<TOps::FFT<5, ND>>(cshape, true);
  auto FP = Ops::Mul<Cx>(Ops::Mul<Cx>(F, P), D);
  auto S = std::make_shared<TOps::TensorScale<Cx, 5, 0, 1>>(cshape, ref);
  auto SFP = Ops::Mul<Cx>(S, FP);

  // Weights (mask)
  auto const           r = CollapseToArray(ref);
  Eigen::ArrayXf const ra = r.abs();
  Eigen::ArrayXf const rm = OtsuMasked(ra);
  auto                 M = std::make_shared<Ops::Mask<Cx>>(rm, nB * nC);
  auto                 MSFP = Ops::Mul<Cx>(M, SFP);

  Cx5 kernels;
  if (λ > 0.f) {
    // Smoothness penalthy (Sobolev Norm, Nonlinear Inversion Paper Uecker 2008)
    CxN<ND> const sw = SobolevWeights<ND>(kW, l).template cast<Cx>();
    auto const    swv = CollapseToConstVector(sw);
    auto          W = std::make_shared<Ops::DiagRep<Cx>>(swv, 1, Product(LastN<5 - ND>(kshape)));
    auto          L = std::make_shared<Ops::DiagScale<Cx>>(W->rows(), λ);

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

template auto
EstimateKernels<2>(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5;
template auto
EstimateKernels<3>(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5;

/*
 * For 2D, assume multislice so only inflate the first two dimensions
 */
template <int ND> auto KernelsToMaps(Cx5 const &kernels, Sz<ND> const mat, float const os) -> Cx5
{
  auto const  kshape = kernels.dimensions();
  auto const  fshape = Concatenate(MulToEven(mat, os), LastN<5 - ND>(kshape));
  auto const  cshape = Concatenate(mat, LastN<5 - ND>(kshape));
  float const scale = std::sqrt(Product(FirstN<ND>(fshape)) / (float)Product(FirstN<ND>(kshape)));
  Log::Print("SENSE", "Kernels {} Full maps {} Cropped maps {} Scale {}", kshape, fshape, cshape, scale);
  TOps::Pad<Cx, 5> P(kshape, fshape);
  TOps::FFT<5, ND> F(fshape, false);
  TOps::Pad<Cx, 5> C(cshape, fshape);
  return C.adjoint(F.adjoint(P.forward(kernels))) * Cx(scale);
}

template auto KernelsToMaps(Cx5 const &, Sz2 const, float const) -> Cx5;
template auto KernelsToMaps(Cx5 const &, Sz3 const, float const) -> Cx5;

template <int ND> auto MapsToKernels(Cx5 const &maps, Sz<ND> const kmat, float const os) -> Cx5
{
  auto const  mshape = maps.dimensions();
  auto const  oshape = Concatenate(MulToEven(FirstN<ND>(mshape), os), LastN<5 - ND>(mshape));
  auto const  kshape = Concatenate(kmat, LastN<5 - ND>(mshape));
  float const scale = std::sqrt(Product(FirstN<ND>(oshape)) / (float)Product(FirstN<ND>(mshape)));
  Log::Print("SENSE", "Map Shape {} Oversampled map shape {} Kernel shape {} Scale {}", mshape, oshape, kshape, scale);
  TOps::Pad<Cx, 5> P(mshape, oshape);
  TOps::FFT<5, 3>  F(oshape, true);
  TOps::Pad<Cx, 5> C(kshape, oshape);
  return C.adjoint(F.adjoint(P.forward(maps))) * Cx(scale);
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
    Cx4 const ref = DimDot<3>(c, c).sqrt();
    kernels = EstimateKernels<ND>(c, ref, opts.kWidth, gopts.osamp, opts.l, opts.λ);
  } else {
    HD5::Reader senseReader(opts.type);
    kernels = senseReader.readTensor<Cx5>(HD5::Keys::Data);
  }
  return kernels;
}

template auto Choose(Opts<2> const &opts, GridOpts<2> const &gopts, TrajectoryN<2> const &traj, Cx5 const &noncart) -> Cx5;
template auto Choose(Opts<3> const &opts, GridOpts<3> const &gopts, TrajectoryN<3> const &traj, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
