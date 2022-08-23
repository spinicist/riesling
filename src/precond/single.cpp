#include "single.hpp"

#include "log.h"
#include "mapping.h"
#include "op/nufft.hpp"
#include "threads.h"

namespace rl {

/*
 * Frank Ong's Preconditioner from https://ieeexplore.ieee.org/document/8906069/
 * (without SENSE maps)
 */
SingleChannel::SingleChannel(Trajectory const &traj)
  : Precond{}
{
  Index const imScale = 2;
  float const osamp = imScale * 1.25;
  auto k = make_kernel("FI7", traj.info().type, osamp);
  auto gridder = rl::make_grid<Cx>(k.get(), Mapping(traj, k.get(), osamp, 32), 1);
  gridder->doNotWeightFrames();
  // Keep more than usual otherwise funky numerical issues
  Sz3 sz{traj.info().matrix[0]*imScale, traj.info().matrix[1]*imScale, traj.info().matrix[2]*imScale};
  NUFFTOp nufft(sz, gridder.get());
  Cx3 W(gridder->outputDimensions());
  pre_.resize(gridder->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  W = nufft.A(nufft.Adj(W));
  float const λ = 1.e-12f; // Regularise the division
  pre_.device(Threads::GlobalDevice()) = (W.inverse().abs() + pre_.constant(λ)) / (pre_.constant(1.f + λ));
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
