#include "recon.hpp"

#include "op/compose.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"

namespace rl {
namespace Recon {

auto SENSE(CoreOpts               &coreOpts,
           GridOpts               &gridOpts,
           Trajectory const       &traj,
           Index const             nSlab,
           TOps::SENSE::Ptr const &sense,
           Basis<Cx> const        &basis) -> TOps::TOp<Cx, 4, 4>::Ptr
{
  auto const shape = sense->mapDimensions();

  TOps::TOp<Cx, 5, 3>::Ptr FT = nullptr;
  if (coreOpts.ndft) {
    if (gridOpts.vcc) { Log::Fail("VCC and NDFT not supported yet"); }
    FT = TOps::NDFT<3>::Make(shape, traj.points(), sense->nChannels(), basis);
  } else {
    auto const nC = sense->nChannels() / (gridOpts.vcc ? 2 : 1);
    FT = TOps::NUFFT<3>::Make(shape, traj, gridOpts, nC, basis);
  }

  auto loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(FT, nSlab);
  auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);

  auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
  auto compose2 = std::make_shared<decltype(TOps::Compose(sense, compose1))>(sense, compose1);
  return compose2;
}

auto Channels(bool const            ndft,
              GridOpts             &gridOpts,
              Trajectory const     &traj,
              Eigen::Array3f const &fov,
              Index const           nC,
              Index const           nSlab,
              Basis<Cx> const      &basis) -> TOps::TOp<Cx, 5, 4>::Ptr
{
  Index const nB = basis.dimension(0);
  auto const  shape = traj.matrixForFOV(fov);

  TOps::TOp<Cx, 5, 3>::Ptr FT = nullptr;
  if (ndft) {
    FT = TOps::NDFT<3>::Make(shape, traj.points(), nC, basis);
  } else {
    FT = TOps::NUFFT<3>::Make(shape, traj, gridOpts, nC, basis);
  }

  std::shared_ptr<TOps::TOp<Cx, 6, 4>> loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(FT, nSlab);
  std::shared_ptr<TOps::TOp<Cx, 5, 6>> slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(AddFront(shape, nC, nB), nSlab);

  std::shared_ptr<TOps::TOp<Cx, 5, 4>> compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
  return compose1;
}

} // namespace Recon
} // namespace rl