#pragma once

#include "grid.hpp"

namespace rl {

template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid_internal(Kernel const *k, Mapping const &m, Index const nC)
{
  return std::make_unique<Grid<IP, TP, Scalar>>(dynamic_cast<SizedKernel<IP, TP> const *>(k), m, nC);
}

template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid_internal(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis)
{
  return std::make_unique<Grid<IP, TP, Scalar>>(dynamic_cast<SizedKernel<IP, TP> const *>(k), m, nC, basis);
}

} // namespace rl
