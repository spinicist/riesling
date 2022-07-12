#pragma once

#include "operator.hpp"

namespace rl {
struct Scaling final : Operator<4, 4>
{
  Scaling(R1 const &scales, Sz3 const &dims);

  Sz4 inputDimensions() const;
  Sz4 outputDimensions() const;

  Cx4 A(Cx4 const &in) const;
  Cx4 Adj(Cx4 const &in) const;
  Cx4 Inv(Cx4 const &in) const;

private:
  R1 scales_;
  Sz3 sz_;
};
}
