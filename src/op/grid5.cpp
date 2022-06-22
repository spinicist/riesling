#include "grid.hpp"

std::unique_ptr<GridBase> make_5_e(Kernel const *k, Mapping const &m, bool const fg, std::shared_ptr<Cx5> ws)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<Grid<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, fg, ws);
  } else {
    return std::make_unique<Grid<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, fg, ws);
  }
}
