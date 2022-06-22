#include "grid-basis.hpp"

std::unique_ptr<GridBase> make_5_b(Kernel const *k, Mapping const &m, R2 const &b, bool const fg, std::shared_ptr<Cx5> ws)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<GridBasis<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, b, fg, ws);
  } else {
    return std::make_unique<GridBasis<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, b, fg, ws);
  }
}
