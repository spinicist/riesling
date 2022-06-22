#include "grid.hpp"

std::unique_ptr<GridBase> make_1_e(Kernel const *k, Mapping const &m, bool const fg, std::shared_ptr<Cx5> ws)
{
  return std::make_unique<Grid<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, fg, ws);
}
