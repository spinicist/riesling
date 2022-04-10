#pragma once

#include "precond.hpp"

struct Scales final : Precond<Cx4>
{
  Scales(R1 const &scales);

  Cx4 apply(Cx4 const &in) const;
  Cx4 inv(Cx4 const &in) const;

private:
  R1 scales_;
};
