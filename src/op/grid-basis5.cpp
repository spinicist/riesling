#include "grid-basis.hpp"

std::unique_ptr<GridBase> make_5_b(Kernel const *k, Mapping const &m, Index const nC, R2 const &b, bool const fg)
{
  if (k->throughPlane() == 1) {
    return std::make_unique<GridBasis<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, nC, b, fg);
  } else {
    return std::make_unique<GridBasis<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, nC, b, fg);
  }
}
