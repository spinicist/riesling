#include "grid-echo.hpp"

std::unique_ptr<GridBase> make_1_e(Kernel const *k, Mapping const &m, bool const fg)
{
  return std::make_unique<GridEcho<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, fg);
}
