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
  float const osamp = 1.25;
  // Do NOT increase this beyond a width 3 kernel.
  // I do not fully understand why but increasing it leads to numerical issues,
  // which I assume might be solved by further oversampling but do not have the 
  // time or memory to investigate further
  auto k = make_kernel("FI5", traj.info().type, osamp);
  auto gridder = rl::make_grid<Cx>(k.get(), Mapping(traj, k.get(), osamp*imScale, 32), 1);
  gridder->doNotWeightFrames();
  // Keep more than usual otherwise funky numerical issues
  Sz3 sz = LastN<3>(gridder->inputDimensions());
  // Sz3 sz{traj.info().matrix[0]*imScale, traj.info().matrix[1]*imScale, traj.info().matrix[2]*imScale};
  NUFFTOp nufft(sz, gridder.get());
  Cx3 W(gridder->outputDimensions());
  pre_.resize(gridder->outputDimensions());
  W.setConstant(Cx(1.f, 0.f));
  W = nufft.A(nufft.Adj(W));
  Log::Tensor(W, "single-W");
  float const scale = 1.f; //std::pow(Product(LastN<3>(sz)), 1.5f) / traj.info().matrix.prod();
  pre_.device(Threads::GlobalDevice()) = W.cast<Cxd>().abs().cast<float>() * scale;
  Log::Tensor(Re2(pre_.chip(0, 0)), "single-pre-inv");
  pre_.device(Threads::GlobalDevice()) = (pre_ > 0.f).select(pre_.inverse(), pre_.constant(1.f));
  Log::Tensor(Re2(pre_.chip(0, 0)), "single-pre");
  float const norm = Norm(pre_);
  if (!std::isfinite(norm)) {
    Log::Fail("Single-channel pre-conditioner norm was not finite ({})", norm);
  } else {
    Log::Print("Single-channel pre-conditioner, norm {}", norm);
  }
}

Cx3 SingleChannel::apply(Cx3 const &in) const
{
  assert(LastN<2>(in.dimensions()) == LastN<2>(pre_.dimensions()));
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.broadcast(Sz3{nC, 1, 1}).cast<Cx>();
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Took {}"), Log::ToNow(start));
  LOG_DEBUG("In norm {}", Norm(in));
  LOG_DEBUG("Out norm {}", Norm(p));
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
