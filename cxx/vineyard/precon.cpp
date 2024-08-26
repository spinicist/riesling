#include "precon.hpp"

#include "fft.hpp"
#include "io/reader.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/ops.hpp"
#include "threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceSingle(Trajectory const &traj, Basis::CPtr basis, bool const vcc, float const bias) -> Re2
{
  Trajectory  newTraj(traj.points() * 2.f, Mul(traj.matrix(), 2), traj.voxelSize() * 2.f);
  float const osamp = 1.25;
  Re2         weights;
  if (vcc) {
    TOps::NUFFT<3, true> nufft(newTraj, "ES5", osamp, 1, basis);
    Cx3                  W(nufft.oshape);
    Log::Print("Starting preconditioner calculation");
    W.setConstant(Cx(1.f, 0.f));
    Cx6 const psf = nufft.adjoint(W);
    Cx6       ones(AddFront(traj.matrix(), psf.dimension(0), psf.dimension(1), psf.dimension(2)));
    ones.setConstant(1. / std::sqrt(psf.dimension(0) * psf.dimension(1)));
    TOps::Pad<Cx, 6> padX(ones.dimensions(), psf.dimensions());
    Cx6              xcorr(padX.oshape);
    xcorr.device(Threads::TensorDevice()) = padX.forward(ones);
    auto const ph = FFT::PhaseShift(LastN<3>(xcorr.dimensions()));
    FFT::Forward(xcorr, Sz3{3, 4, 5}, ph);
    xcorr.device(Threads::TensorDevice()) = xcorr * xcorr.conjugate();
    FFT::Adjoint(xcorr, Sz3{3, 4, 5}, ph);
    xcorr.device(Threads::TensorDevice()) = xcorr * psf;
    weights = nufft.forward(xcorr).abs().chip(0, 0);
    // I do not understand this scaling factor but it's in Frank's code and works
    float scale =
      std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(LastN<3>(ones.dimensions()));
    weights.device(Threads::TensorDevice()) = ((weights * scale) + bias).inverse();
  } else {
    TOps::NUFFT<3, false> nufft(newTraj, "ES5", osamp, 1, basis);
    Cx3                   W(nufft.oshape);
    Log::Print("Starting preconditioner calculation");
    W.setConstant(Cx(1.f, 0.f));
    Cx5 const psf = nufft.adjoint(W);
    Cx5       ones(AddFront(traj.matrix(), psf.dimension(0), psf.dimension(1)));
    ones.setConstant(1.f);
    TOps::Pad<Cx, 5> padX(ones.dimensions(), psf.dimensions());
    Cx5              xcorr(padX.oshape);
    xcorr.device(Threads::TensorDevice()) = padX.forward(ones);
    auto const ph = FFT::PhaseShift(LastN<3>(xcorr.dimensions()));
    FFT::Forward(xcorr, Sz3{2, 3, 4}, ph);
    xcorr.device(Threads::TensorDevice()) = xcorr * xcorr.conjugate();
    FFT::Adjoint(xcorr, Sz3{2, 3, 4}, ph);
    xcorr.device(Threads::TensorDevice()) = xcorr * psf;
    weights = nufft.forward(xcorr).abs().chip(0, 0);
    // I do not understand this scaling factor but it's in Frank's code and works
    float scale =
      std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(LastN<3>(ones.dimensions()));
    weights.device(Threads::TensorDevice()) = (weights * scale) + bias;
  }

  float const norm = Norm(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

auto MakeKspacePre(Trajectory const  &traj,
                   Index const        nC,
                   Index const        nT,
                   Basis::CPtr        basis,
                   std::string const &type,
                   float const        bias,
                   bool const         ndft) -> std::shared_ptr<Ops::Op<Cx>>
{
  if (type == "" || type == "none") {
    Log::Print("Using no preconditioning");
    return std::make_shared<Ops::Identity<Cx>>(nC * traj.nSamples() * traj.nTraces() * nT);
  } else if (type == "kspace") {
    if (ndft) { Log::Warn("Preconditioning for NDFT is not supported yet, using NUFFT preconditioner"); }
    Re2 const              w = KSpaceSingle(traj, basis, false, bias);
    Eigen::VectorXcf const wv = CollapseToArray(w);
    return std::make_shared<Ops::DiagRep<Cx>>(wv, nC, nT);
  } else {
    HD5::Reader reader(type);
    Re2         w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(0) != traj.nSamples() || w.dimension(1) != traj.nTraces()) {
      Log::Fail("Preconditioner dimensions on disk {} did not match trajectory {}x{}", w.dimension(0), w.dimension(1),
                traj.nSamples(), traj.nTraces());
    }
    Eigen::VectorXcf const wv = CollapseToArray(w);
    return std::make_shared<Ops::DiagRep<Cx>>(wv, nC, nT);
  }
}

} // namespace rl
