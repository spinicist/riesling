#include "grid.hpp"

std::unique_ptr<GridBase> make_1(Kernel const *k, Mapping const &m, Index const nC)
{
  return std::make_unique<Grid<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, nC);
}
