#pragma once

#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/trajectory.hpp"
#include "rl/types.hpp"

namespace rl {

struct SliceNCT
{
  Cx5 ks;
  Re3 tp;
};

auto SliceNC(Sz3 const   channel,
             Sz3 const   sample,
             Sz3 const   trace,
             Sz3 const   slab,
             Sz3 const   time,
             Index const tps,
             Sz3 const   segment,
             Cx5 const  &ks,
             Re3 const  &trajPoints) -> SliceNCT;

} // namespace rl
