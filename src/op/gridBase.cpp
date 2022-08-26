#include "gridBase.hpp"
#include "grid.hpp"
#include "io/reader.hpp"
#include "newkernel.hpp"

namespace rl {

template <typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid(
  Trajectory const &traj, std::string const kType, float const osamp, Index const nC, std::optional<Re2> const &basis)
{

  Info const &info = traj.info();

  Index const nZ = info.matrix[2] / info.slabs;

  if (nZ == 1) {
    if (kType == "FI3") {
      return std::make_unique<Grid<Scalar, NewFlatIron<2, 3>>>(traj, osamp, nC, basis);
    } else if (kType == "FI4") {
      return std::make_unique<Grid<Scalar, NewFlatIron<2, 4>>>(traj, osamp, nC, basis);
    } else if (kType == "FI5") {
      return std::make_unique<Grid<Scalar, NewFlatIron<2, 5>>>(traj, osamp, nC, basis);
    } else if (kType == "FI7") {
      return std::make_unique<Grid<Scalar, NewFlatIron<2, 7>>>(traj, osamp, nC, basis);
    }
  } else {
    if (kType == "FI3") {
      return std::make_unique<Grid<Scalar, NewFlatIron<3, 3>>>(traj, osamp, nC, basis);
    } else if (kType == "FI4") {
      return std::make_unique<Grid<Scalar, NewFlatIron<3, 4>>>(traj, osamp, nC, basis);
    } else if (kType == "FI5") {
      return std::make_unique<Grid<Scalar, NewFlatIron<3, 5>>>(traj, osamp, nC, basis);
    } else if (kType == "FI7") {
      return std::make_unique<Grid<Scalar, NewFlatIron<3, 7>>>(traj, osamp, nC, basis);
    }
  }
  Log::Fail("Invalid kernel type {}", kType);
}

template std::unique_ptr<GridBase<Cx>>
make_grid<Cx>(Trajectory const &, std::string const, float const, Index const, std::optional<Re2> const &);
template std::unique_ptr<GridBase<float>>
make_grid<float>(Trajectory const &, std::string const, float const, Index const, std::optional<Re2> const &);

} // namespace rl
