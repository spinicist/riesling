#include "single.hpp"

#include "log.hpp"
#include "mapping.hpp"
#include "op/nufft.hpp"
#include "threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
SingleChannel::SingleChannel(Trajectory const &traj)
  : Precond{}
{
  Log::Print<Log::Level::High>("Single Channel Pre-conditioner start");
  Info const info = traj.info();
  Info newInfo = info;
  std::transform(
    newInfo.matrix.begin(), newInfo.matrix.end(), newInfo.matrix.begin(), [](Index const i) { return i * 2; });
  Trajectory newTraj(newInfo, traj.points(), traj.frames());
  float const osamp = 1.25;
  auto gridder = rl::make_grid<Cx, 3>(newTraj, "ES5", osamp, 1);
  gridder->doNotWeightnFrames();
  // Keep more than usual otherwise funky numerical issues
  // Sz3 sz = LastN<3>(gridder->inputDimensions());
  NUFFTOp nufft(newInfo.matrix, gridder.get());
  Cx3 W(gridder->outputDimensions());
  pre_.resize(gridder->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft.adjoint(W);

  Cx5 ones(AddFront(info.matrix, 1, 1));
  ones.setConstant(1.f);
  // I do not understand this scaling factor but it's in Frank's code and works
  float const scale = std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(info.matrix) / Norm2(ones);
  PadOp<5> padX(ones.dimensions(), LastN<3>(psf.dimensions()));
  FFTOp<5> fftX(psf.dimensions());

  Cx5 xcorr = fftX.adjoint(fftX.forward(padX.forward(ones)).abs().square().cast<Cx>());
  pre_.device(Threads::GlobalDevice()) = nufft.forward(psf * xcorr).abs() * pre_.constant(scale);
  pre_.device(Threads::GlobalDevice()) = (pre_ > 0.f).select(pre_.inverse(), pre_.constant(1.f));
  Log::Tensor(Re2(pre_.chip(0, 0)), "single-pre");
  float const norm = Norm(pre_);
  if (!std::isfinite(norm)) {
    Log::Fail("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {}", norm);
  }
}

Cx3 SingleChannel::apply(Cx3 const &in) const
{
  assert(LastN<2>(in.dimensions()) == LastN<2>(pre_.dimensions()));
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.broadcast(Sz3{nC, 1, 1}).cast<Cx>();
  LOG_DEBUG(FMT_STRING("Single-channel preconditioner Norm {}->{}. Took {}"), Norm(in), Norm(p), Log::ToNow(start));
  return p;
}

} // namespace rl
