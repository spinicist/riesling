#pragma once

#include "operator.hpp"

namespace rl {

struct SDCOp final : Operator<3, 3>
{
  SDCOp(Re2 const &dc, Index const nc);
  SDCOp(Sz2 const &dims, Index const nc);

  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

  auto adjoint(Cx3 const &x) const -> Input;

private:
  Sz3 dims_;
  Re2 dc_;
};

} // namespace rl
