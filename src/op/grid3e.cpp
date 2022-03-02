#include "grid-echo.hpp"

std::unique_ptr<GridBase> make_3_e(Kernel const *k, Mapping const &m, bool const fg)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<GridEcho<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, fg);
  } else {
    return std::make_unique<GridEcho<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, fg);
  }
}
