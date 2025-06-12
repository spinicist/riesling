#pragma once

#include "../op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx>
struct StackProx final : Prox<Scalar>
{
  using Vector = Prox<Scalar>::Vector;
  using Map = Prox<Scalar>::Map;
  using CMap = Prox<Scalar>::CMap;
  using Ptr = std::shared_ptr<StackProx>;

  StackProx(std::vector<std::shared_ptr<Prox<Scalar>>> p);
  StackProx(std::shared_ptr<Prox<Scalar>> p1, std::vector<std::shared_ptr<Prox<Scalar>>> const p);

  using Prox<Scalar>::apply;

  void apply(float const α, CMap x, Map z) const;
  void apply(std::shared_ptr<Ops::Op<Scalar>> const α, CMap x, Map z) const;

private:
  std::vector<std::shared_ptr<Prox<Scalar>>> proxs;
};

} // namespace rl::Proxs
