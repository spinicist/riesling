#include "grid.h"
#include "grid-basis.hpp"
#include "grid-echo.hpp"

#include "../tensorOps.h"
#include "../threads.h"

std::unique_ptr<GridBase> make_grid(Kernel const *k, Mapping const &m, bool const fg, Log &log)
{
  switch (k->inPlane()) {
  case 1:
    return std::make_unique<GridEcho<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, fg, log);
  case 3:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridEcho<3, 1>>(
        dynamic_cast<SizedKernel<3, 1> const *>(k), m, fg, log);
    case 3:
      return std::make_unique<GridEcho<3, 3>>(
        dynamic_cast<SizedKernel<3, 3> const *>(k), m, fg, log);
    }
    __builtin_unreachable();
  case 5:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridEcho<5, 1>>(
        dynamic_cast<SizedKernel<5, 1> const *>(k), m, fg, log);
    case 5:
      return std::make_unique<GridEcho<5, 5>>(
        dynamic_cast<SizedKernel<5, 5> const *>(k), m, fg, log);
    }
    __builtin_unreachable();
  }
  __builtin_unreachable();
}

std::unique_ptr<GridBase>
make_grid_basis(Kernel const *k, Mapping const &m, R2 const &b, bool const fg, Log &log)
{
  switch (k->inPlane()) {
  case 1:
    return std::make_unique<GridBasis<1, 1>>(
      dynamic_cast<SizedKernel<1, 1> const *>(k), m, b, fg, log);
  case 3:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridBasis<3, 1>>(
        dynamic_cast<SizedKernel<3, 1> const *>(k), m, b, fg, log);
    case 3:
      return std::make_unique<GridBasis<3, 3>>(
        dynamic_cast<SizedKernel<3, 3> const *>(k), m, b, fg, log);
    }
    __builtin_unreachable();
  case 5:
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<GridBasis<5, 1>>(
        dynamic_cast<SizedKernel<5, 1> const *>(k), m, b, fg, log);
    case 5:
      return std::make_unique<GridBasis<5, 5>>(
        dynamic_cast<SizedKernel<5, 5> const *>(k), m, b, fg, log);
    }
    __builtin_unreachable();
  }
  __builtin_unreachable();
}
