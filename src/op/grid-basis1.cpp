#include "grid.hpp"

std::unique_ptr<GridBase> make_1_b(Kernel const *k, Mapping const &m, Index const nC, R2 const &b)
{
  return std::make_unique<Grid<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, nC, b);
}
