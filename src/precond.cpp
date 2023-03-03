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
auto KSpaceSingle(Trajectory const &traj, std::optional<Re2> const &basis, float const bias) -> Re2
{
  Log::Print<Log::Level::High>("Ong's Single-channer preconditioner");
  Info const info = traj.info();
  Info newInfo = info;
  std::transform(
    newInfo.matrix.begin(), newInfo.matrix.begin() + traj.nDims(), newInfo.matrix.begin(), [](Index const i) { return i * 2; });
  Trajectory newTraj(newInfo, traj.points());
  float const osamp = 1.25;
  auto nufft = make_nufft(newTraj, "ES5", osamp, 1, newTraj.matrix(), basis);
  Cx4 W(nufft->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Log::Tensor(psf, "single-psf");
  Cx5 ones(AddFront(info.matrix, psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1. / std::sqrt(psf.dimension(1)));
  PadOp<Cx, 5, 3> padX(info.matrix, LastN<3>(psf.dimensions()), FirstN<2>(psf.dimensions()));
  FFTOp<5, 3> fftX(psf.dimensions());
  Cx5 xcorr = fftX.cforward(padX.cforward(ones)).abs().square().cast<Cx>();
  xcorr = fftX.adjoint(xcorr);
  xcorr = xcorr * psf;
  Log::Tensor(Cx5(xcorr), "pre-img");
  Re2 weights = nufft->forward(xcorr).abs().chip(0, 3).chip(0, 0);
  // I do not understand this scaling factor but it's in Frank's code and works
  float scale =
    std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(info.matrix) / Product(LastN<3>(ones.dimensions()));
  Log::Tensor(weights, "pre-weights");
  weights.device(Threads::GlobalDevice()) = ((weights * scale) + bias).inverse();
  Log::Tensor(weights, "precond");
  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

std::shared_ptr<Functor1<Cx4>>
make_pre(std::string const &type, Trajectory const &traj, std::optional<Re2> const &basis, float const bias)
{
  if (type == "" || type == "none") {
    Log::Print(FMT_STRING("Using no preconditioning"));
    return std::make_shared<IdentityProx<Cx4>>();
  } else if (type == "kspace") {
    return std::make_shared<BroadcastPower<Cx, 4, 1, 1>>(
      KSpaceSingle(traj, basis, bias).cast<Cx>(), "KSpace Preconditioner");
  } else {
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
    return std::make_shared<BroadcastPower<Cx, 4, 1, 1>>(pre.cast<Cx>());
  }
}

} // namespace rl
