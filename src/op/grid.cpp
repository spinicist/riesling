#include "grid.h"
#include "grid-basis.hpp"
#include "grid-echo.hpp"

#include "../tensorOps.h"
#include "../threads.h"

std::unique_ptr<GridBase> make_grid(Kernel const *k, Mapping const &m, bool const fg)
{
  switch (k->inPlane()) {
  case 1:
    return std::make_unique<GridEcho<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, fg);
  case 3:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridEcho<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, fg);
    case 3:
      return std::make_unique<GridEcho<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, fg);
    }
    __builtin_unreachable();
  case 5:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridEcho<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, fg);
    case 5:
      return std::make_unique<GridEcho<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, fg);
    }
    __builtin_unreachable();
  }
  __builtin_unreachable();
}

std::unique_ptr<GridBase>
make_grid_basis(Kernel const *k, Mapping const &m, R2 const &b, bool const fg)
{
  switch (k->inPlane()) {
  case 1:
    return std::make_unique<GridBasis<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, b, fg);
  case 3:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridBasis<3, 1>>(
        dynamic_cast<SizedKernel<3, 1> const *>(k), m, b, fg);
    case 3:
      return std::make_unique<GridBasis<3, 3>>(
        dynamic_cast<SizedKernel<3, 3> const *>(k), m, b, fg);
    }
    __builtin_unreachable();
  case 5:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridBasis<5, 1>>(
        dynamic_cast<SizedKernel<5, 1> const *>(k), m, b, fg);
    case 5:
      return std::make_unique<GridBasis<5, 5>>(
        dynamic_cast<SizedKernel<5, 5> const *>(k), m, b, fg);
    }
    __builtin_unreachable();
  }
  __builtin_unreachable();
}
