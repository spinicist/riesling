#pragma once

#include "op/ops.hpp"
#include "prox/prox.hpp"

namespace rl {

struct ADMM
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  std::shared_ptr<Op> A; // Op for least-squares
  std::shared_ptr<Op> M; // Pre-conditioner
  Index               lsqLimit = 8;
  float               aTol = 1.e-6f;
  float               bTol = 1.e-6f;
  float               cTol = 1.e-6f;

  std::vector<std::shared_ptr<Op>>              reg_ops;
  std::vector<std::shared_ptr<Proxs::Prox<Cx>>> prox;
  Index                                         outerLimit = 8;
  float                                         ε = 1.e-3f;

  std::function<void(Index const, Vector const &)>                    debug_x = nullptr;
  std::function<void(Index const, Index const, ADMM::Vector const &)> debug_z = nullptr;

  auto run(Cx const *bdata, float ρ) const -> Vector;
};

} // namespace rl
