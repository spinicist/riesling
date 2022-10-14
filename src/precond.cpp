#include "precond.hpp"

#include "func/multiply.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/nufft.hpp"
#include "threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceSingle(Trajectory const &traj, std::optional<Re2> const basis) -> Re2
{
  Log::Print<Log::Level::High>("Single Channel Pre-conditioner start");
  Info const info = traj.info();
  Info newInfo = info;
  std::transform(
    newInfo.matrix.begin(), newInfo.matrix.begin() + traj.nDims(), newInfo.matrix.begin(), [](Index const i) { return i * 2; });
  Trajectory newTraj(newInfo, traj.points(), traj.frames());
  float const osamp = 1.25;
  auto nufft = make_nufft(newTraj, "ES5", osamp, 1, newTraj.matrix(), nullptr, basis);
  Cx4 W(nufft->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Log::Tensor(psf, "single-psf");
  Cx5 ones(AddFront(info.matrix, psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1.f);
  // I do not understand this scaling factor but it's in Frank's code and works
  float const scale = std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(info.matrix) / Norm2(ones);
  PadOp<5, 3> padX(info.matrix, LastN<3>(psf.dimensions()), FirstN<2>(psf.dimensions()));
  FFTOp<5, 3> fftX(psf.dimensions());

  Cx5 xcorr = fftX.adjoint(fftX.forward(padX.forward(ones)).abs().square().cast<Cx>());
  Log::Tensor(xcorr, "single-xcorr");
  Re4 weights = nufft->forward(psf * xcorr).abs();
  weights.device(Threads::GlobalDevice()) = (weights > 0.f).select((weights * scale).inverse(), weights.constant(1.f));
  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Fail("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {}", norm);
  }
  return weights.chip(0, 3).chip(0, 0);
}

std::unique_ptr<Functor<Cx4>> make_pre(std::string const &type, Trajectory const &traj)
{
  if (type == "" || type == "none") {
    Log::Print(FMT_STRING("Using no preconditioning"));
    return std::make_unique<Identity<Cx4>>();
  } else if (type == "kspace") {
    Log::Print(FMT_STRING("Using Ong's k-space preconditioner"));
    return std::make_unique<BroadcastMultiply<Cx, 4, 1, 1>>(KSpaceSingle(traj).cast<Cx>());
  } else {
    Log::Print(FMT_STRING("Using preconditioner from: {}"), type);
    HD5::Reader reader(type);
    Re2 pre = reader.readTensor<Re2>(HD5::Keys::Precond);
    if (pre.dimension(0) != traj.nSamples() || pre.dimension(1) != traj.nTraces()) {
      Log::Fail(
        FMT_STRING("Preconditioner dimensions on disk {} did not match trajectory {}x{}"),
        pre.dimension(0),
        pre.dimension(1),
        traj.nSamples(),
        traj.nTraces());
    }
    return std::make_unique<BroadcastMultiply<Cx, 4, 1, 1>>(pre.cast<Cx>());
  }
}

} // namespace rl
