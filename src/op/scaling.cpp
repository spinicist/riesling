#include "scaling.hpp"

#include "../log.h"
#include "../tensorOps.h"

namespace rl {

Scaling::Scaling(R1 const &scales, Sz3 const &sz)
  : sz_{sz}
{
  scales_ = scales;
  Log::Print(FMT_STRING("SCALING Created with weights: {}"), Transpose(scales_));
}

Sz4 Scaling::inputDimensions() const
{
  return AddFront(sz_, scales_.dimension(0));
}

Sz4 Scaling::outputDimensions() const
{
  return AddFront(sz_, scales_.dimension(0));
}

Cx4 Scaling::Adj(Cx4 const &in) const
{
  auto const start = Log::Now();
  auto const sz = in.dimensions();
  Cx4 p(sz);
  p.device(Threads::GlobalDevice()) =
    in / scales_.cast<Cx>().reshape(Sz4{sz[0], 1, 1, 1}).broadcast(Sz4{1, sz[1], sz[2], sz[3]});
  Log::Debug(FMT_STRING("SCALING Adjoint Took {}"), Log::ToNow(start));
  return p;
}

Cx4 Scaling::A(Cx4 const &in) const
{
  auto const start = Log::Now();
  auto const sz = in.dimensions();
  Cx4 p(sz);
  p.device(Threads::GlobalDevice()) =
    in / scales_.cast<Cx>().reshape(Sz4{sz[0], 1, 1, 1}).broadcast(Sz4{1, sz[1], sz[2], sz[3]});
  Log::Debug(FMT_STRING("SCALING Took {}"), Log::ToNow(start));
  return p;
}

Cx4 Scaling::Inv(Cx4 const &in) const
{
  auto const start = Log::Now();
  auto const sz = in.dimensions();
  Cx4 p(sz);
  p.device(Threads::GlobalDevice()) =
    in * scales_.cast<Cx>().reshape(Sz4{sz[0], 1, 1, 1}).broadcast(Sz4{1, sz[1], sz[2], sz[3]});
  Log::Debug(FMT_STRING("SCALES Inverse Took {}"), Log::ToNow(start));
  return p;
}
} // namespace rl
