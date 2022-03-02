#include "grid-basis.hpp"

std::unique_ptr<GridBase> make_3_b(Kernel const *k, Mapping const &m, R2 const &b, bool const fg)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<GridBasis<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, b, fg);
  } else {
    return std::make_unique<GridBasis<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, b, fg);
  }
}
