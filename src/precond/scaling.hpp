#pragma once

#include "precond.hpp"

namespace rl {
struct Scaling final : Precond<Cx4>
{
  Scaling(Sz4 const &dims);
  Scaling(Sz4 const &dims, Re1 const &scales);

  void setScales(Re1 const &scales);

  Cx4 apply(Cx4 const &in) const;
  Cx4 inv(Cx4 const &in) const;

private:
  Sz4 sz_;
  Re1 scales_;  
};
}
