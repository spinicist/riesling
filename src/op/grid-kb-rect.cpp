#include "grid.hpp"
#include "io/reader.hpp"
#include "kernel/kaiser.hpp"
#include "kernel/rectilinear.hpp"
#include "make_grid.hpp"

namespace rl {

template <typename Scalar, size_t ND>
auto make_kb_rect(Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Scalar, ND>>
{
  if (W == 3) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, KaiserBessel<3>>>>(
      Mapping<ND>(traj, osamp, Rectilinear<ND, KaiserBessel<3>>::PadWidth), nC, basis);
  } else if (W == 4) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, KaiserBessel<4>>>>(
      Mapping<ND>(traj, osamp, Rectilinear<ND, KaiserBessel<4>>::PadWidth), nC, basis);
  } else if (W == 5) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, KaiserBessel<5>>>>(
      Mapping<ND>(traj, osamp, Rectilinear<ND, KaiserBessel<5>>::PadWidth), nC, basis);
  } else if (W == 7) {
    return std::make_unique<Grid<Scalar, Rectilinear<ND, KaiserBessel<7>>>>(
      Mapping<ND>(traj, osamp, Rectilinear<ND, KaiserBessel<7>>::PadWidth), nC, basis);
  }
  Log::Fail("Invalid kernel width {}", W);
}

template auto
make_kb_rect<Cx, 2>(Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Cx, 2>>;
template auto
make_kb_rect<Cx, 3>(Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<Cx, 3>>;
template auto make_kb_rect<float, 2>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<float, 2>>;
template auto make_kb_rect<float, 3>(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, std::optional<Re2> const &basis)
  -> std::unique_ptr<GridBase<float, 3>>;

} // namespace rl
