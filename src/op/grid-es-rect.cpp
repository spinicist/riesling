#include "grid.hpp"
#include "gridBase.hpp"
#include "io/reader.hpp"
#include "kernel/expsemi.hpp"
#include "kernel/rectilinear.hpp"

namespace rl {

template <typename Scalar, size_t ND>
auto make_es_rect(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Scalar, ND>>
{
  if (W == 3) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, ExpSemi<3>>>>(traj, osamp, nC, basis);
  } else if (W == 4) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, ExpSemi<4>>>>(traj, osamp, nC, basis);
  } else if (W == 5) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, ExpSemi<5>>>>(traj, osamp, nC, basis);
  } else if (W == 7) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, ExpSemi<7>>>>(traj, osamp, nC, basis);
  }
  Log::Fail("Invalid kernel width {}", W);
}

template auto make_es_rect<Cx, 2>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Cx, 2>>;
template auto make_es_rect<Cx, 3>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Cx, 3>>;
template auto make_es_rect<float, 2>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<float, 2>>;
template auto make_es_rect<float, 3>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<float, 3>>;

} // namespace rl
