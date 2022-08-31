#include "single.hpp"

#include "log.h"
#include "mapping.h"
#include "op/nufft.hpp"
#include "threads.h"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
SingleChannel::SingleChannel(Trajectory const &traj)
  : Precond{}
{
  Info const info = traj.info();
  Info newInfo = info;
  std::transform(
    newInfo.matrix.begin(), newInfo.matrix.end(), newInfo.matrix.begin(), [](Index const i) { return i * 2; });
  Trajectory newTraj(newInfo, traj.points(), traj.frames());
  float const osamp = 1.25;
  auto gridder = rl::make_grid<Cx, 3>(newTraj, "ES3", osamp, 1);
  gridder->doNotWeightFrames();
  // Keep more than usual otherwise funky numerical issues
  // Sz3 sz = LastN<3>(gridder->inputDimensions());
  NUFFTOp nufft(newInfo.matrix, gridder.get());
  Cx3 W(gridder->outputDimensions());
  pre_.resize(gridder->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft.adjoint(W);

  Cx5 ones(AddFront(info.matrix, 1, 1));
  ones.setConstant(1.f);
  PadOp<5> padX(ones.dimensions(), psf.dimensions());
  FFTOp<5> fftX(psf.dimensions());
  
  Cx5 xcorr = fftX.adjoint(fftX.forward(padX.forward(ones)).abs().square().cast<Cx>());
  Log::Tensor(xcorr, "single-xcorr");
  W = nufft.forward(psf * xcorr);
  Log::Tensor(W, "single-W");
  float const scale = 1.f; // std::pow(Product(LastN<3>(sz)), 1.5f) / traj.info().matrix.prod();
  pre_.device(Threads::GlobalDevice()) = W.abs() * scale;
  Log::Tensor(Re2(pre_.chip(0, 0)), "single-pre-inv");
  pre_.device(Threads::GlobalDevice()) = (pre_ > 0.f).select(pre_.inverse(), pre_.constant(1.f));
  Log::Tensor(Re2(pre_.chip(0, 0)), "single-pre");
  float const norm = Norm(pre_);
  if (!std::isfinite(norm)) {
    Log::Fail("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner, norm {}", norm);
  }
}

Cx3 SingleChannel::apply(Cx3 const &in) const
{
  assert(LastN<2>(in.dimensions()) == LastN<2>(pre_.dimensions()));
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.broadcast(Sz3{nC, 1, 1}).cast<Cx>();
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Took {}"), Log::ToNow(start));
  LOG_DEBUG("In norm {}", Norm(in));
  LOG_DEBUG("Out norm {}", Norm(p));
  return p;
}

Cx3 SingleChannel::inv(Cx3 const &in) const
{
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.broadcast(Sz3{nC, 1, 1}).cast<Cx>();
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Inverse Took {}"), Log::ToNow(start));
  return p;
}

} // namespace rl
