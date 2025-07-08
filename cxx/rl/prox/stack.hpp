#pragma once

#include "../op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

struct Stack final : Prox
{
  PROX_INHERIT
  Stack(std::vector<Ptr> p);
  Stack(Ptr p1, std::vector<Ptr> const ps);
  static auto Make(std::vector<Ptr> p) -> Ptr;

  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  std::vector<Ptr> proxs;
};

} // namespace rl::Proxs
