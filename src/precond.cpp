#include "precond.hpp"

#include "io/reader.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/ops.hpp"
#include "threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceSingle(Trajectory const &traj, Basis<Cx> const &basis, float const bias, bool const ndft) -> Re2
{
  Info const info = traj.info();
  Info       newInfo = info;
  std::transform(newInfo.matrix.begin(), newInfo.matrix.begin() + traj.nDims(), newInfo.matrix.begin(),
                 [](Index const i) { return i * 2; });
  Trajectory  newTraj(newInfo, traj.points());
  float const osamp = 1.25;
  auto        nufft = ndft ? make_ndft(newTraj.points(), 1, newTraj.matrix(), basis)
                           : make_nufft(newTraj, "ES5", osamp, 1, newTraj.matrix(), basis);
  Cx4         W(nufft->oshape);
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Cx5       ones(AddFront(info.matrix, psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1. / std::sqrt(psf.dimension(0) * psf.dimension(1)));
  PadOp<Cx, 5, 3> padX(info.matrix, LastN<3>(psf.dimensions()), FirstN<2>(psf.dimensions()));
  auto            fftX = FFT::Make<5, 3>(psf.dimensions());
  Cx5             xcorr = padX.forward(ones);
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

auto make_kspace_pre(
  std::string const &type, Index const nC, Trajectory const &traj, Basis<Cx> const &basis, float const bias, bool const ndft)
  -> std::shared_ptr<Ops::Op<Cx>>
{
  if (type == "" || type == "none") {
    Log::Print("Using no preconditioning");
    return std::make_shared<Ops::Identity<Cx>>(nC * traj.nSamples() * traj.nTraces());
  } else if (type == "kspace") {
    Re2 const              w = KSpaceSingle(traj, basis, bias, ndft);
    Eigen::VectorXcf const wv = CollapseToArray(w);
    return std::make_shared<Ops::DiagRep<Cx>>(nC, wv);
  } else {
    HD5::Reader reader(type);
    Re2         w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(0) != traj.nSamples() || w.dimension(1) != traj.nTraces()) {
      Log::Fail("Preconditioner dimensions on disk {} did not match trajectory {}x{}", w.dimension(0), w.dimension(1),
                traj.nSamples(), traj.nTraces());
    }
    Eigen::VectorXcf const wv = CollapseToArray(w);
    return std::make_shared<Ops::DiagRep<Cx>>(nC, wv);
  }
}

} // namespace rl
