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
auto KSpaceSingle(rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, float const λ) -> Re2
{
  Log::Print("Precon", "Starting preconditioner calculation");
  Trajectory newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  auto       nufft = TOps::NUFFT<3>::Make(gridOpts, newTraj, 1, nullptr);
  Cx3        W(nufft->oshape);
  W.setConstant(Cx(1.f, 0.f));
  Cx5 const psf = nufft->adjoint(W);
  Cx5       ones(AddFront(traj.matrix(), psf.dimension(0), psf.dimension(1)));
  ones.setConstant(1.f);
  TOps::Pad<Cx, 5> padX(ones.dimensions(), psf.dimensions());
  Cx5              xcor(padX.oshape);
  xcor.device(Threads::TensorDevice()) = padX.forward(ones);
  FFT::Forward(xcor, Sz3{2, 3, 4});
  xcor.device(Threads::TensorDevice()) = xcor * xcor.conjugate();
  FFT::Adjoint(xcor, Sz3{2, 3, 4});
  xcor.device(Threads::TensorDevice()) = xcor * psf;
  Re3 weights = nufft->forward(xcor).abs();
  // I do not understand this scaling factor but it's in Frank's code and works
  float scale =
    std::pow(Product(LastN<3>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(LastN<3>(ones.dimensions()));
  weights.device(Threads::TensorDevice()) = (1.f + λ) / ((weights * scale) + λ);

  float const norm = Norm<true>(weights);
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
auto KSpaceMulti(Cx5 const &smaps, rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, float const λ) -> Re3
{
  Log::Print("Precon", "Calculating multichannel-preconditioner");
  Trajectory  newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  Index const nC = smaps.dimension(1);
  Index const nSamp = traj.nSamples();
  Index const nTrace = traj.nTraces();
  Re3         weights(nC, nSamp, nTrace);

  auto      nufft = TOps::NUFFT<3>(gridOpts, newTraj, 1, nullptr);
  Sz5 const psfShape = nufft.ishape;
  Sz5 const smapShape = smaps.dimensions();
  if (smapShape[0] > 1 && smapShape[0] != psfShape[0]) {
    throw Log::Failure("Precon", "SENSE maps had basis dimension {}, expected {}", smapShape[0], psfShape[0]);
  }
  Cx3 W(nufft.oshape);
  Cx5 psf(psfShape);
  W.setConstant(Cx(1.f, 0.f));
  nufft.adjoint(W, psf);

  // I do not understand this scaling factor but it's in Frank's code and works
  float scale = std::pow(Product(LastN<3>(psfShape)), 1.5f) / Product(traj.matrix());
  Log::Print("Precon", "Map shape {} scale {}", smapShape, scale);

  Sz5 const smap1Shape = AddFront(LastN<3>(smapShape), smapShape[0], 1);
  Sz5 const xcor1Shape = AddFront(LastN<3>(psfShape), smapShape[0], 1);

  auto padXC = TOps::Pad<Cx, 5>(smap1Shape, xcor1Shape);
  Cx5  smap1(smap1Shape), xcorTemp(xcor1Shape), xcor1(xcor1Shape), xcor(psfShape);
  for (Index si = 0; si < nC; si++) {
    float const ni = Norm2<true>(smaps.chip<1>(si));
    xcor1.setZero();
    for (Index sj = 0; sj < nC; sj++) {
      // Log::Print("Precon", "Cross-correlation channel {}-{}", si, sj);
      smap1.device(Threads::TensorDevice()) =
        smaps.slice(Sz5{0, si, 0, 0, 0}, smap1Shape) * smaps.slice(Sz5{0, sj, 0, 0, 0}, smap1Shape).conjugate();
      padXC.forward(smap1, xcorTemp);
      FFT::Forward(xcorTemp, Sz3{2, 3, 4});
      xcor1.device(Threads::TensorDevice()) += xcorTemp * xcorTemp.conjugate();
    }
    Log::Print("Precon", "Channel {} map squared norm {}", si, ni);
    FFT::Adjoint(xcor1, Sz3{2, 3, 4});
    if (smapShape[0] == 1) {
      xcor.device(Threads::TensorDevice()) = xcor1.broadcast(Sz5{psfShape[0], 1, 1, 1, 1}) * psf;
    } else {
      xcor.device(Threads::TensorDevice()) = xcor1 * psf;
    }
    weights.slice(Sz3{si, 0, 0}, Sz3{1, nSamp, nTrace}).device(Threads::TensorDevice()) =
      (1.f + λ) / (nufft.forward(xcor).abs() * scale / ni + λ);
  }
  float const norm = Norm<true>(weights);
  if (!std::isfinite(norm)) {
    Log::Print("Precon", "Pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Precon", "Pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

auto LoadKSpacePrecon(std::string const &fname, Trajectory const &traj, Sz5 const shape) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  HD5::Reader reader(fname);
  Index const o = reader.order(HD5::Keys::Weights);
  if (o == 2) {
    Re2 const w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(1) != traj.nSamples() || w.dimension(2) != traj.nTraces()) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(1),
                         w.dimension(2), traj.nSamples(), traj.nTraces());
    }
    return std::make_shared<TOps::TensorScale<Cx, 5, 1, 2>>(shape, w.cast<Cx>());
  } else if (o == 3) {
    Re3 const w = reader.readTensor<Re3>(HD5::Keys::Weights);
    if (w.dimension(0) != shape[1]) {
      throw Log::Failure("Precon", "Preconditioner on disk had {} channels, expected {}", w.dimension(0), shape[1]);
    }
    if (w.dimension(1) != traj.nSamples() || w.dimension(2) != traj.nTraces()) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(1),
                         w.dimension(2), traj.nSamples(), traj.nTraces());
    }
    return std::make_shared<TOps::TensorScale<Cx, 5, 0, 2>>(shape, w.cast<Cx>());
  } else {
    throw Log::Failure("Precon", "On-disk weights had order {}, expected 2 or 3", o);
  }
}

auto MakeKSpaceSingle(PreconOpts const              &opts,
                      rl::TOps::Grid<3>::Opts const &gridOpts,
                      Trajectory const              &traj,
                      Index const                    nC,
                      Index const                    nS,
                      Index const                    nT) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  Sz5 const shape{nC, traj.nSamples(), traj.nTraces(), nS, nT};
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return std::make_shared<TOps::Identity<Cx, 5>>(shape);
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.λ);
    return std::make_shared<TOps::TensorScale<Cx, 5, 1, 2>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    throw Log::Failure("Precon", "Multichannel preconditioner requested without SENSE maps");
  } else {
    return LoadKSpacePrecon(opts.type, traj, shape);
  }
}

auto MakeKSpaceMulti(PreconOpts const              &opts,
                     rl::TOps::Grid<3>::Opts const &gridOpts,
                     Trajectory const              &traj,
                     Cx5 const                     &smaps,
                     Index const                    nS,
                     Index const                    nT) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  Sz5 const shape{smaps.dimension(1), traj.nSamples(), traj.nTraces(), nS, nT};
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return std::make_shared<TOps::Identity<Cx, 5>>(shape);
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.λ);
    return std::make_shared<TOps::TensorScale<Cx, 5, 1, 2>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    Re3 const w = KSpaceMulti(smaps, gridOpts, traj, opts.λ);
    return std::make_shared<TOps::TensorScale<Cx, 5, 0, 2>>(shape, w.cast<Cx>());
  } else {
    return LoadKSpacePrecon(opts.type, traj, shape);
  }
}

} // namespace rl
