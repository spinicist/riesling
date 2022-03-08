#include "grid-basis.hpp"

std::unique_ptr<GridBase> make_1_b(Kernel const *k, Mapping const &m, R2 const &b, bool const fg)
{
  return std::make_unique<GridBasis<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, b, fg);
}
