#include "scaling.hpp"

#include "../log.h"
#include "../tensorOps.h"

namespace rl {

Scaling::Scaling(Sz4 const &sz)
  : sz_{sz}
  , scales_{Re1(sz_[0])}
{
  scales_.setConstant(1.f);
  Log::Print(FMT_STRING("SCALING Created with weights: {}"), Transpose(scales_));
}

Scaling::Scaling(Sz4 const &sz, Re1 const &scales)
  : sz_{sz}
  , scales_{scales}
{
  assert(scales_.dimension(0) != sz_[0]);
  Log::Print(FMT_STRING("SCALING Created with weights: {}"), Transpose(scales_));
}

void Scaling::setScales(Re1 const &scales)
{
  assert(scales.dimension(0) != sz_[0]);
  Log::Print(FMT_STRING("Set scaling to: {}"), Transpose(scales_));
  scales_ = scales;
}

Cx4 Scaling::apply(Cx4 const &in) const
{
  auto const start = Log::Now();
  assert(in.dimensions() == sz_);
  Cx4 p(sz_);
  p.device(Threads::GlobalDevice()) =
    in / scales_.cast<Cx>().reshape(Sz4{sz_[0], 1, 1, 1}).broadcast(Sz4{1, sz_[1], sz_[2], sz_[3]});
  Log::Debug(FMT_STRING("SCALING Took {}"), Log::ToNow(start));
  return p;
}

Cx4 Scaling::inv(Cx4 const &in) const
{
  auto const start = Log::Now();
  assert(in.dimensions() == sz_);
  Cx4 p(sz_);
  p.device(Threads::GlobalDevice()) =
    in * scales_.cast<Cx>().reshape(Sz4{sz_[0], 1, 1, 1}).broadcast(Sz4{1, sz_[1], sz_[2], sz_[3]});
  Log::Debug(FMT_STRING("SCALING Inverse Took {}"), Log::ToNow(start));
  LOG_DEBUG("Norm {}", Norm(p));
  return p;
}

} // namespace rl
