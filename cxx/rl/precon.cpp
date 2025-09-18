#include "precon.hpp"

#include "fft.hpp"
#include "io/reader.hpp"
#include "log/log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/tensorscale.hpp"
#include "op/top-id.hpp"
#include "sys/threads.hpp"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 *
 * This is the version without SENSE maps. It still needs the auto-correlation step in the middle, I think to remove the effect
 * of the cropping during the NUFFT. I also tested simply grid adj * grid, which gave reasonable results but would do a double
 * convolution with the gridding kernel.
 */
template <int ND>
auto KSpaceSingle(GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, float const max, Basis::CPtr basis) -> Re2
{
  Log::Print("Precon", "Starting preconditioner calculation");
  TrajectoryN<ND> newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  auto            nufft = TOps::MakeNUFFT<ND>(gridOpts, newTraj, 1, basis);
  Cx3             W(nufft->oshape);
  W.setConstant(Cx(1.f, 0.f));
  CxN<ND + 2> const psf = nufft->adjoint(W);
  CxN<ND + 2>       ones(AddBack(traj.matrix(), psf.dimension(ND), psf.dimension(ND + 1)));
  ones.setConstant(1.f);
  TOps::Pad<ND + 2> padX(ones.dimensions(), psf.dimensions());
  CxN<ND + 2>       xcor(padX.oshape);
  xcor.device(Threads::TensorDevice()) = padX.forward(ones);
  FFT::Forward(xcor, FirstN<ND>(Sz3{0, 1, 2}));
  xcor.device(Threads::TensorDevice()) = xcor * xcor.conjugate();
  FFT::Adjoint(xcor, FirstN<ND>(Sz3{0, 1, 2}));
  xcor.device(Threads::TensorDevice()) = xcor * psf;
  // I do not understand this scaling factor but it's in Frank's code and works
  float scale =
    std::pow(Product(FirstN<ND>(psf.dimensions())), 1.5f) / Product(traj.matrix()) / Product(FirstN<ND>(ones.dimensions()));
  Re3 weights = nufft->forward(xcor).abs() * scale;
  Log::Print("Precon", "Thresholding to {}", max);
  weights.device(Threads::TensorDevice()) = (weights < 1.f / max).select(weights.constant(max), 1.f / weights);

  float const norm = Norm<true>(weights);
  if (!std::isfinite(norm)) {
    throw Log::Failure("Precon", "Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Precon", "Single-channel pre-conditioner finished, norm {} scale {} min {} max {}", norm, scale,
               Minimum(weights), Maximum(weights));
  }
  return weights.chip<0>(0);
}

template auto KSpaceSingle(GridOpts<2> const &, TrajectoryN<2> const &, float const, Basis::CPtr) -> Re2;
template auto KSpaceSingle(GridOpts<3> const &, TrajectoryN<3> const &, float const, Basis::CPtr) -> Re2;

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
auto KSpaceMulti(Cx5 const &smaps, GridOpts<3> const &gridOpts, Trajectory const &traj, float const λ, Basis::CPtr basis) -> Re3
{
  Log::Print("Precon", "Calculating multichannel-preconditioner");
  Trajectory  newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  Index const nC = smaps.dimension(4);
  Index const nSamp = traj.nSamples();
  Index const nTrace = traj.nTraces();
  Re3         weights(nC, nSamp, nTrace);

  auto        nufft = TOps::NUFFT<3>(gridOpts, newTraj, 1, basis);
  Sz5 const   psfShape = nufft.ishape;
  Sz5 const   smapShape = smaps.dimensions();
  Index const nB = smapShape[3];
  if (nB > 1 && nB != psfShape[3]) {
    throw Log::Failure("Precon", "SENSE maps had basis dimension {}, expected {}", nB, psfShape[3]);
  }
  Cx3 W(nufft.oshape);
  Cx5 psf(psfShape);
  W.setConstant(Cx(1.f, 0.f));
  nufft.adjoint(W, psf);

  // I do not understand this scaling factor but it's in Frank's code and works
  float scale = std::pow(Product(FirstN<3>(psfShape)), 1.5f) / Product(traj.matrix());
  Log::Print("Precon", "Map shape {} scale {}", smapShape, scale);

  Sz5 const smap1Shape = AddBack(FirstN<3>(smapShape), nB, 1);
  Sz5 const xcor1Shape = AddBack(FirstN<3>(psfShape), nB, 1);

  auto padXC = TOps::Pad<5>(smap1Shape, xcor1Shape);
  Cx5  smap1(smap1Shape), xcorTemp(xcor1Shape), xcor1(xcor1Shape), xcor(psfShape);
  for (Index si = 0; si < nC; si++) {
    float const ni = Norm2<true>(smaps.chip<1>(si));
    xcor1.setZero();
    for (Index sj = 0; sj < nC; sj++) {
      // Log::Print("Precon", "Cross-correlation channel {}-{}", si, sj);
      smap1.device(Threads::TensorDevice()) =
        smaps.slice(Sz5{0, 0, 0, 0, si}, smap1Shape) * smaps.slice(Sz5{0, 0, 0, 0, sj}, smap1Shape).conjugate();
      padXC.forward(smap1, xcorTemp);
      FFT::Forward(xcorTemp, Sz3{0, 1, 2});
      xcor1.device(Threads::TensorDevice()) += xcorTemp * xcorTemp.conjugate();
    }
    Log::Print("Precon", "Channel {} map squared norm {}", si, ni);
    FFT::Adjoint(xcor1, Sz3{0, 1, 2});
    if (nB == 1 && nB != psfShape[3]) {
      xcor.device(Threads::TensorDevice()) = xcor1.broadcast(Sz5{1, 1, 1, psfShape[3], 1}) * psf;
    } else {
      xcor.device(Threads::TensorDevice()) = xcor1 * psf;
    }
    weights.slice(Sz3{si, 0, 0}, Sz3{1, nSamp, nTrace}).device(Threads::TensorDevice()) =
      (nufft.forward(xcor).abs() * scale / ni + λ) / (1.f + λ);
  }
  float const norm = Norm<true>(weights);
  if (!std::isfinite(norm)) {
    throw Log::Failure("Precon", "Pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Precon", "Pre-conditioner finished, norm {} min {} max {}", norm, Minimum(weights), Maximum(weights));
  }
  return weights;
}

template <int NB> auto LoadKSpacePrecon(std::string const &fname, Index const nSamp, Index const nTrace, Sz<3 + NB> const shape)
  -> TOps::TOp<3 + NB>::Ptr
{
  HD5::Reader reader(fname);
  Index const o = reader.order(HD5::Keys::Weights);
  if (o == 2) {
    Re2 const w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(0) != nSamp || w.dimension(1) != nTrace) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(0),
                         w.dimension(1), nSamp, nTrace);
    }
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (o == 3) {
    Re3 const w = reader.readTensor<Re3>(HD5::Keys::Weights);
    if (w.dimension(0) != shape[1]) {
      throw Log::Failure("Precon", "Preconditioner on disk had {} channels, expected {}", w.dimension(0), shape[1]);
    }
    if (w.dimension(1) != nSamp || w.dimension(2) != nTrace) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(1),
                         w.dimension(2), nSamp, nTrace);
    }
    return std::make_shared<TOps::TensorScale<3 + NB, 0, NB>>(shape, w.cast<Cx>());
  } else {
    throw Log::Failure("Precon", "On-disk weights had order {}, expected 2 or 3", o);
  }
}

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Index const            nC,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB>::Ptr
{
  auto const shape = Concatenate(Sz3{nC, traj.nSamples(), traj.nTraces()}, bshape);
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return nullptr;
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.max);
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    throw Log::Failure("Precon", "Multichannel preconditioner requested without SENSE maps");
  } else {
    return LoadKSpacePrecon<NB>(opts.type, traj.nSamples(), traj.nTraces(), shape);
  }
}

template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Index const, Sz0 const)
  -> TOps::TOp<3>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Index const, Sz1 const)
  -> TOps::TOp<4>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Index const, Sz2 const)
  -> TOps::TOp<5>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Index const, Sz0 const)
  -> TOps::TOp<3>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Index const, Sz1 const)
  -> TOps::TOp<4>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Index const, Sz2 const)
  -> TOps::TOp<5>::Ptr;

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Cx5 const             &smaps,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB, 3 + NB>::Ptr
{
  Index const nC = smaps.dimension(4);
  auto const  shape = Concatenate(Sz3{nC, traj.nSamples(), traj.nTraces()}, bshape);
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return nullptr;
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.max);
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    Re3 const w = KSpaceMulti(smaps, gridOpts, traj, opts.max);
    return std::make_shared<TOps::TensorScale<3 + NB, 0, NB>>(shape, w.cast<Cx>());
  } else {
    return LoadKSpacePrecon<NB>(opts.type, traj.nSamples(), traj.nTraces(), shape);
  }
}

template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Cx5 const &, Sz<0> const)
  -> TOps::TOp<3, 3>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Cx5 const &, Sz<1> const)
  -> TOps::TOp<4, 4>::Ptr;
template auto MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Cx5 const &, Sz<2> const)
  -> TOps::TOp<5, 5>::Ptr;

} // namespace rl
