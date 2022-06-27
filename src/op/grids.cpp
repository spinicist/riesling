#include "grids.h"

#include "io/reader.hpp"

// Forward declarations
std::unique_ptr<GridBase> make_1(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_1_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);
std::unique_ptr<GridBase> make_3(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_3_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);
std::unique_ptr<GridBase> make_5(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_5_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);

std::unique_ptr<GridBase> make_grid(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile)
{
  if (basisFile.size()) {
    HD5::Reader basisReader(basisFile);
    R2 const b = basisReader.readTensor<R2>(HD5::Keys::Basis);
    switch (k->inPlane()) {
    case 1:
      return make_1_b(k, m, nC, b);
    case 3:
      return make_3_b(k, m, nC, b);
    case 5:
      return make_5_b(k, m, nC, b);
    }
  } else {
    switch (k->inPlane()) {
    case 1:
      return make_1(k, m, nC);
    case 3:
      return make_3(k, m, nC);
    case 5:
      return make_5(k, m, nC);
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}
