#include "precond.hpp"

#include "log.hpp"
#include "mapping.hpp"
#include "op/nufft.hpp"
#include "op/tensorscale.hpp"
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
  Cx4 W(nufft->oshape);
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Cx5 ones(AddFront(info.matrix, psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1. / std::sqrt(psf.dimension(0) * psf.dimension(1)));
  PadOp<Cx, 5, 3> padX(info.matrix, LastN<3>(psf.dimensions()), FirstN<2>(psf.dimensions()));
  auto fftX = FFT::Make<5, 3>(psf.dimensions());
  Cx5 xcorr = padX.forward(ones);
  fftX->forward(xcorr);
  xcorr = xcorr * xcorr.conjugate();
  fftX->reverse(xcorr);
  xcorr = xcorr * psf;
  Re2 weights = nufft->forward(xcorr).abs().chip(0, 3).chip(0, 0);
  // I do not understand this scaling factor but it's in Frank's code and works
  float scale =
    std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(info.matrix) / Product(LastN<3>(ones.dimensions()));
  weights.device(Threads::GlobalDevice()) = ((weights * scale) + bias).inverse();
  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

std::shared_ptr<TensorOperator<Cx, 4>>
make_pre(std::string const &type, Sz4 const dims, Trajectory const &traj, std::optional<Re2> const &basis, float const bias)
{
  if (type == "" || type == "none") {
    Log::Print("Using no preconditioning");
    return std::make_shared<TensorIdentity<Cx, 4>>(dims);
  } else if (type == "kspace") {
    return std::make_shared<TensorScale<Cx, 4, 1, 1>>(dims, KSpaceSingle(traj, basis, bias).cast<Cx>());
  } else {
    HD5::Reader reader(type);
    Re2 pre = reader.readTensor<Re2>(HD5::Keys::Precond);
    if (pre.dimension(0) != traj.nSamples() || pre.dimension(1) != traj.nTraces()) {
      Log::Fail(
        "Preconditioner dimensions on disk {} did not match trajectory {}x{}",
        pre.dimension(0),
        pre.dimension(1),
        traj.nSamples(),
        traj.nTraces());
    }
    return std::make_shared<TensorScale<Cx, 4, 1, 1>>(dims, pre.cast<Cx>());
  }
}

} // namespace rl
