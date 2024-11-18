#include "precon.hpp"

#include "fft.hpp"
#include "io/reader.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/tensorscale.hpp"
#include "sys/threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceSingle(rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, Basis::CPtr basis, float const λ) -> Re2
{
  Log::Print("Precon", "Starting preconditioner calculation");
  Trajectory newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  auto       nufft = TOps::NUFFT<3>::Make(gridOpts, newTraj, 1, basis);
  Cx3        W(nufft->oshape);
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Cx5       ones(AddFront(traj.matrix(), psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1.f);
  TOps::Pad<Cx, 5> padX(ones.dimensions(), psf.dimensions());
  Cx5              xcorr(padX.oshape);
  xcorr.device(Threads::TensorDevice()) = padX.forward(ones);
  FFT::Forward(xcorr, Sz3{2, 3, 4});
  xcorr.device(Threads::TensorDevice()) = xcorr * xcorr.conjugate();
  FFT::Adjoint(xcorr, Sz3{2, 3, 4});
  xcorr.device(Threads::TensorDevice()) = xcorr * psf;
  Re3 weights = nufft->forward(xcorr).abs();
  // I do not understand this scaling factor but it's in Frank's code and works
  float scale =
    std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(LastN<3>(ones.dimensions()));
  weights.device(Threads::TensorDevice()) = (1.f + λ) / ((weights * scale) + λ);

  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Precon", "Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Precon", "Single-channel pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights),
               Maximum(weights));
  }
  return weights.chip<0>(0);
}

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceMulti(
  Cx5 const &smaps, rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, Basis::CPtr basis, float const λ) -> Re3
{
  Log::Print("Precon", "Starting preconditioner calculation");
  Trajectory  newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  Index const nC = smaps.dimension(1);
  Index const nSamp = traj.nSamples();
  Index const nTrace = traj.nTraces();
  Re3         weights(nC, nSamp, nTrace);

  auto nufft = TOps::NUFFT<3>(gridOpts, newTraj, 1, basis);
  Cx3  W(nufft.oshape);
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft.adjoint(W);

  Sz5 smapShape = smaps.dimensions();
  smapShape[1] = 1;
  TOps::Pad<Cx, 5> padXC(smapShape, nufft.ishape);
  Cx5              xcorrChan(nufft.ishape);
  for (Index si = 0; si < nC; si++) {
    float const ni = Norm2(smaps.chip<1>(si));
    xcorrChan.setZero();
    for (Index sj = 0; sj < nC; sj++) {
      Cx5 xcorrImage = smaps.slice(Sz5{0, si, 0, 0, 0}, smapShape) * smaps.slice(Sz5{0, sj, 0, 0, 0}, smapShape).conjugate();
      Cx5 xcorr = padXC.forward(xcorrImage);
      FFT::Forward(xcorr, Sz3{2, 3, 4});
      xcorrChan.device(Threads::TensorDevice()) += xcorr * xcorr.conjugate();
    }
    FFT::Adjoint(xcorrChan, Sz3{2, 3, 4});
    xcorrChan.device(Threads::TensorDevice()) = xcorrChan * psf;
    weights.slice(Sz3{si, 0, 0}, Sz3{1, nSamp, nTrace}) = nufft.forward(xcorrChan).abs();
  }

  // I do not understand this scaling factor but it's in Frank's code and works
  float scale = std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(LastN<3>(smapShape));
  weights.device(Threads::TensorDevice()) = (1.f + λ) / ((weights * scale) + λ);

  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Precon", "Pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Precon", "Pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

auto MakeKSpaceSingle(PreconOpts const              &opts,
                      rl::TOps::Grid<3>::Opts const &gridOpts,
                      Trajectory const              &traj,
                      Index const                    nC,
                      Index const                    nS,
                      Index const                    nT,
                      Basis::CPtr                    basis) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  Sz5 const shape{nC, traj.nSamples(), traj.nTraces(), nS, nT};
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return std::make_shared<TOps::Identity<Cx, 5>>(shape);
  } else if (opts.type == "kspace") {
    Re2 const w = KSpaceSingle(gridOpts, traj, basis, opts.λ);
    return std::make_shared<TOps::TensorScale<Cx, 5, 1, 2>>(shape, w.cast<Cx>());
  } else {
    HD5::Reader reader(opts.type);
    Re2 const   w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(1) != traj.nSamples() || w.dimension(2) != traj.nTraces()) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {} did not match trajectory {}x{}", w.dimension(1),
                         w.dimension(2), traj.nSamples(), traj.nTraces());
    }
    return std::make_shared<TOps::TensorScale<Cx, 5, 1, 2>>(shape, w.cast<Cx>());
  }
}

} // namespace rl
