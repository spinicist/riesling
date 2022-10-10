#include "pre-kspace.hpp"

#include "log.hpp"
#include "mapping.hpp"
#include "op/nufft.hpp"
#include "threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
KSpaceSingle::KSpaceSingle(Trajectory const &traj)
  : Functor<Cx4>()
{
  Log::Print<Log::Level::High>("Single Channel Pre-conditioner start");
  Info const info = traj.info();
  Info newInfo = info;
  std::transform(
    newInfo.matrix.begin(), newInfo.matrix.begin() + traj.nDims(), newInfo.matrix.begin(), [](Index const i) {
      return i * 2;
    });
  Trajectory newTraj(newInfo, traj.points(), traj.frames());
  float const osamp = 1.25;
  auto nufft = make_nufft(newTraj, "ES5", osamp, 1);
  Cx4 W(nufft->outputDimensions());
  weights.resize(nufft->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Log::Tensor(psf, "single-psf");
  Cx5 ones(AddFront(info.matrix, 1, 1));
  ones.setConstant(1.f);
  // I do not understand this scaling factor but it's in Frank's code and works
  float const scale = std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(info.matrix) / Norm2(ones);
  PadOp<5, 3> padX(ones.dimensions(), LastN<3>(psf.dimensions()));
  FFTOp<5, 3> fftX(psf.dimensions());

  Cx5 xcorr = fftX.adjoint(fftX.forward(padX.forward(ones)).abs().square().cast<Cx>());
  Log::Tensor(xcorr, "single-xcorr");
  weights.device(Threads::GlobalDevice()) = nufft->forward(psf * xcorr).abs() * weights.constant(scale);
  weights.device(Threads::GlobalDevice()) = (weights > 0.f).select(weights.inverse(), weights.constant(1.f));
  Log::Tensor(Re2(weights.chip(0, 3).chip(0, 0)), "single-pre");
  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Fail("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {}", norm);
  }
}

auto KSpaceSingle::operator()(Cx4 const &in) const -> Cx4
{
  assert(LastN<3>(in.dimensions()) == LastN<3>(weights.dimensions()));
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx4 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * weights.broadcast(Sz4{nC, 1, 1, 1}).cast<Cx>();
  LOG_DEBUG(FMT_STRING("Single-channel preconditioner Norm {}->{}. Took {}"), Norm(in), Norm(p), Log::ToNow(start));
  return p;
}

} // namespace rl
