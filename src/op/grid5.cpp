#include "grid.hpp"

std::unique_ptr<GridBase> make_5(Kernel const *k, Mapping const &m, Index const nC)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<Grid<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, nC);
  } else {
    return std::make_unique<Grid<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, nC);
  }
}
