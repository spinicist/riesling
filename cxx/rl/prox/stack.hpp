#pragma once

#include "../op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx>
struct StackProx final : Prox<Scalar>
{
  PROX_INHERIT(Scalar)
  StackProx(std::vector<std::shared_ptr<Prox<Scalar>>> p);
  StackProx(std::shared_ptr<Prox<Scalar>> p1, std::vector<std::shared_ptr<Prox<Scalar>>> const p);

  void primal(float const α, CMap x, Map z) const;
  void dual(float const α, CMap x, Map z) const;

private:
  std::vector<std::shared_ptr<Prox<Scalar>>> proxs;
};

} // namespace rl::Proxs
