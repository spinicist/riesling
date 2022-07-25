#include "single.hpp"

#include "log.h"
#include "mapping.h"
#include "op/nufft.hpp"
#include "threads.h"

namespace rl {

SingleChannel::SingleChannel(Trajectory const &traj, Kernel const *k)
  : Precond{}
{
  auto gridder = rl::make_grid<Cx>(k, Mapping(traj, k, 4.f, 32), 1);
  NUFFTOp nufft(LastN<3>(gridder->inputDimensions()), gridder.get());
  Cx3 W(AddFront(LastN<2>(gridder->outputDimensions()), 1));
  W.setConstant(1.f);
  W = nufft.A(nufft.Adj(W));
  pre_ = (W.real() > 0.f).select(W.real().inverse(), W.real().constant(0.f));
  Log::Debug("SINGLE-CHANNEL Created");
  Log::Tensor(pre_, "pre");
}

Cx3 SingleChannel::apply(Cx3 const &in) const
{
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.cast<Cx>().broadcast(Sz3{nC, 1, 1});
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Took {}"), Log::ToNow(start));
  return p;
}

Cx3 SingleChannel::inv(Cx3 const &in) const
{
  auto const start = Log::Now();
  Index const nC = in.dimension(0);
  Cx3 p(in.dimensions());
  p.device(Threads::GlobalDevice()) = in * pre_.cast<Cx>().broadcast(Sz3{nC, 1, 1});
  Log::Debug(FMT_STRING("SINGLE-CHANNEL Inverse Took {}"), Log::ToNow(start));
  return p;
}

}
