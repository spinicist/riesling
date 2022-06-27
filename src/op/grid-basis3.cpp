#include "grid.hpp"

std::unique_ptr<GridBase> make_3_b(Kernel const *k, Mapping const &m, Index const nC, R2 const &b)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<Grid<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, nC, b);
  } else {
    return std::make_unique<Grid<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, nC, b);
  }
}
