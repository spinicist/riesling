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
    if (k->throughPlane() == 1) {
      return std::make_unique<GridEcho<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, fg);
    } else {
      return std::make_unique<GridEcho<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, fg);
    }
  case 4:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridEcho<4, 1>>(dynamic_cast<SizedKernel<4, 1> const *>(k), m, fg);
    } else {
      return std::make_unique<GridEcho<4, 4>>(dynamic_cast<SizedKernel<4, 4> const *>(k), m, fg);
    }
  case 5:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridEcho<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, fg);
    } else {
      return std::make_unique<GridEcho<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, fg);
    }
  case 6:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridEcho<6, 1>>(dynamic_cast<SizedKernel<6, 1> const *>(k), m, fg);
    } else {
      return std::make_unique<GridEcho<6, 6>>(dynamic_cast<SizedKernel<6, 6> const *>(k), m, fg);
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

std::unique_ptr<GridBase>
make_grid_basis(Kernel const *k, Mapping const &m, R2 const &b, bool const fg)
{
  switch (k->inPlane()) {
  case 1:
    return std::make_unique<GridBasis<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, b, fg);
  case 3:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridBasis<3, 1>>(
        dynamic_cast<SizedKernel<3, 1> const *>(k), m, b, fg);
    } else {
      return std::make_unique<GridBasis<3, 3>>(
        dynamic_cast<SizedKernel<3, 3> const *>(k), m, b, fg);
    }
  case 4:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridBasis<4, 1>>(
        dynamic_cast<SizedKernel<4, 1> const *>(k), m, b, fg);
    } else {
      return std::make_unique<GridBasis<4, 4>>(
        dynamic_cast<SizedKernel<4, 4> const *>(k), m, b, fg);
    }
  case 5:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridBasis<5, 1>>(
        dynamic_cast<SizedKernel<5, 1> const *>(k), m, b, fg);
    } else {
      return std::make_unique<GridBasis<5, 5>>(
        dynamic_cast<SizedKernel<5, 5> const *>(k), m, b, fg);
    }
  case 6:
    if (k->throughPlane() == 1) {
      return std::make_unique<GridBasis<6, 1>>(
        dynamic_cast<SizedKernel<6, 1> const *>(k), m, b, fg);
    } else {
      return std::make_unique<GridBasis<6, 6>>(
        dynamic_cast<SizedKernel<6, 6> const *>(k), m, b, fg);
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}
