#pragma once

#include "operator.h"

struct GridOp final : Operator<4, 3>
{
  GridOp();

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  Input::Dimensions inSize() const;
  Output::Dimensions outSize() const;

private:
  // std::vector<Mapping> coords_;
  // std::vector<int32_t> sortedIndices_;
  // Sz3 dims_;
  // float DCexp_;
  // Kernel *kernel_;
  // bool safe_;
  // Log &log_;
};
