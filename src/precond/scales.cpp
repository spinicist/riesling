#include "scales.hpp"

#include "../log.h"
#include "../tensorOps.h"

Scales::Scales(R1 const &scales)
  : Precond{}
{
  scales_ = scales.pow(0.25f);
  scales_ = scales_ / scales_(0);
  Log::Print(FMT_STRING("SCALES Created with weights: {}"), Transpose(scales_));
}

Cx4 Scales::apply(Cx4 const &in) const
{
  auto const start = Log::Now();
  auto const sz = in.dimensions();
  Cx4 p(sz);
  p.device(Threads::GlobalDevice()) =
    in * scales_.cast<Cx>().reshape(Sz4{sz[0], 1, 1, 1}).broadcast(Sz4{1, sz[1], sz[2], sz[3]});
  Log::Debug(FMT_STRING("SCALES Took {}"), Log::ToNow(start));
  return p;
}

Cx4 Scales::inv(Cx4 const &in) const
{
  auto const start = Log::Now();
  auto const sz = in.dimensions();
  Cx4 p(sz);
  p.device(Threads::GlobalDevice()) =
    in / scales_.cast<Cx>().reshape(Sz4{sz[0], 1, 1, 1}).broadcast(Sz4{1, sz[1], sz[2], sz[3]});
  Log::Debug(FMT_STRING("SCALES Inverse Took {}"), Log::ToNow(start));
  return p;
}