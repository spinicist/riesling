#include "single.hpp"

#include "log.h"
#include "mapping.h"
#include "op/nufft.hpp"
#include "threads.h"

namespace rl {

SingleChannel::SingleChannel(Trajectory const &traj, Kernel const *k, std::optional<Re2> const &basis)
  : Precond{}
{
  float const os = 2.1f;
  auto gridder = rl::make_grid<Cx>(k, Mapping(traj, k, os, 32), 1, basis);
  gridder->doNotWeightFrames();
  // Crop out the janky corners
  Sz3 sz{traj.info().matrix[0], traj.info().matrix[1], traj.info().matrix[2]};
  NUFFTOp nufft(sz, gridder.get());
  Cx3 W(gridder->outputDimensions());
  pre_.resize(gridder->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  W = nufft.A(nufft.Adj(W));
  pre_.device(Threads::GlobalDevice()) = (W.abs() > 0.f).select(W.abs().inverse(), pre_.constant(1.f));
  Log::Tensor(pre_, "single-pre");
}

Cx3 SingleChannel::apply(Cx3 const &in) const
{
  assert(LastN<2>(in.dimensions()) == LastN<2>(pre_.dimensions()));
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.broadcast(Sz3{nC, 1, 1}).cast<Cx>();
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Took {}"), Log::ToNow(start));
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
