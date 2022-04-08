#include "grids.h"

// Forward declarations
std::unique_ptr<GridBase> make_1_e(Kernel const *, Mapping const &, bool const);
std::unique_ptr<GridBase> make_3_e(Kernel const *, Mapping const &, bool const);
std::unique_ptr<GridBase> make_5_e(Kernel const *, Mapping const &, bool const);

std::unique_ptr<GridBase> make_grid(Kernel const *k, Mapping const &m, bool const fg)
{
  switch (k->inPlane()) {
  case 1:
    return make_1_e(k, m, fg);
  case 3:
    return make_3_e(k, m, fg);
  case 5:
    return make_5_e(k, m, fg);
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

// More forward declarations
std::unique_ptr<GridBase> make_1_b(Kernel const *, Mapping const &, R2 const &b, bool const);
std::unique_ptr<GridBase> make_3_b(Kernel const *, Mapping const &, R2 const &b, bool const);
std::unique_ptr<GridBase> make_5_b(Kernel const *, Mapping const &, R2 const &b, bool const);

std::unique_ptr<GridBase>
make_grid_basis(Kernel const *k, Mapping const &m, R2 const &b, bool const fg)
{
  switch (k->inPlane()) {
  case 1:
    return make_1_b(k, m, b, fg);
  case 3:
    return make_3_b(k, m, b, fg);
  case 5:
    return make_5_b(k, m, b, fg);
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}