#include "gridBase.hpp"
#include "io/reader.hpp"

namespace rl {

// Forward Declare
template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid_internal(Kernel const *k, Mapping const &m, Index const nC);
template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>>
make_grid_internal(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

template <typename Scalar>
std::unique_ptr<GridBase<Scalar>>
make_grid(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile)
{
  if (!basisFile.empty()) {
    HD5::Reader basisReader(basisFile);
    R2 const b = basisReader.readTensor<R2>(HD5::Keys::Basis);
    switch (k->inPlane()) {
    case 1:
      return make_grid_internal<1, 1, Scalar>(k, m, nC, b);
    case 3:
      if (k->throughPlane() == 1) {
        return make_grid_internal<3, 1, Scalar>(k, m, nC, b);
      } else if (k->throughPlane() == 3) {
        return make_grid_internal<3, 3, Scalar>(k, m, nC, b);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_grid_internal<5, 1, Scalar>(k, m, nC, b);
      } else if (k->throughPlane() == 5) {
        return make_grid_internal<5, 5, Scalar>(k, m, nC, b);
      }
      break;
    }
  } else {
    switch (k->inPlane()) {
    case 1:
      return make_grid_internal<1, 1, Scalar>(k, m, nC);
    case 3:
      if (k->throughPlane() == 1) {
        return make_grid_internal<3, 1, Scalar>(k, m, nC);
      } else if (k->throughPlane() == 3) {
        return make_grid_internal<3, 3, Scalar>(k, m, nC);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_grid_internal<5, 1, Scalar>(k, m, nC);
      } else if (k->throughPlane() == 5) {
        return make_grid_internal<5, 5, Scalar>(k, m, nC);
      }
      break;
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

template std::unique_ptr<GridBase<float>>
make_grid<float>(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile);
template std::unique_ptr<GridBase<Cx>>
make_grid<Cx>(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile);

} // namespace rl
