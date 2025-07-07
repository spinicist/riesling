#pragma once

#include "../op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx>
struct Stack final : Prox<Scalar>
{
  PROX_INHERIT(Scalar)
  Stack(std::vector<Ptr> p);
  Stack(Ptr p1, std::vector<Ptr> const ps);
  static auto Make(std::vector<Ptr> p) -> Ptr;

  void primal(float const α, CMap x, Map z) const;
  void dual(float const α, CMap x, Map z) const;

private:
  std::vector<Ptr> proxs;
};

} // namespace rl::Proxs
