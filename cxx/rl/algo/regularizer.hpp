#pragma once

#include "../op/top.hpp"
#include "../prox/prox.hpp"

#include <variant>

namespace rl {

struct Regularizer
{
  using SizeN = std::variant<Sz4, Sz5, Sz6>;
  Ops::Op::Ptr     T;
  Proxs::Prox::Ptr P;
  SizeN                shape;
};

} // namespace rl
