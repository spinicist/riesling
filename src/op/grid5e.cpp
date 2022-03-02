#include "grid-echo.hpp"

std::unique_ptr<GridBase> make_5_e(Kernel const *k, Mapping const &m, bool const fg)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<GridEcho<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, fg);
  } else {
    return std::make_unique<GridEcho<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, fg);
  }
}
