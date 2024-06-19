#include "recon.hpp"

#include "op/compose.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/reshape.hpp"
#include "op/sense.hpp"

namespace rl {
namespace Recon {

auto SENSE(CoreOpts         &coreOpts,
           GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Basis<Cx> const  &basis,
           Cx5 const        &data) -> TOps::TOp<Cx, 4, 4>::Ptr
{
  if (coreOpts.ndft) {
    if (gridOpts.vcc) { Log::Warn("VCC and NDFT not supported yet"); }
    auto sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
    auto nufft = TOps::NDFT<3>::Make(sense->mapDimensions(), traj.points(), sense->nChannels(), basis);
    auto loop = std::make_shared<TOps::Loop<TOps::NDFT<3>>>(nufft, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
    auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
    auto compose2 = std::make_shared<decltype(TOps::Compose(sense, compose1))>(sense, compose1);
    return compose2;
  } else {
    if (gridOpts.vcc) {
      auto sense = std::make_shared<TOps::VCCSENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
      auto nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, sense->nChannels(), basis, sense->mapDimensions());
      auto loop = std::make_shared<TOps::Loop<TOps::NUFFT<3, true>>>(nufft, nSlab);
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 6>>(sense->oshape, nSlab);
      auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
      auto compose2 = std::make_shared<decltype(TOps::Compose(sense, compose1))>(sense, compose1);
      return compose2;
    } else {
      auto sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
      auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, sense->nChannels(), basis, sense->mapDimensions());
      auto loop = std::make_shared<TOps::Loop<TOps::NUFFT<3, false>>>(nufft, nSlab);
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
      auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
      auto compose2 = std::make_shared<decltype(TOps::Compose(sense, compose1))>(sense, compose1);
      return compose2;
    }
  }
}

auto Channels(bool const            ndft,
              GridOpts             &gridOpts,
              Trajectory const     &traj,
              Eigen::Array3f const &fov,
              Index const           nC,
              Index const           nSlab,
              Basis<Cx> const      &basis) -> TOps::TOp<Cx, 5, 4>::Ptr
{
  auto const shape = traj.matrixForFOV(fov);

  if (ndft) {
    auto                                 FT = TOps::NDFT<3>::Make(shape, traj.points(), nC, basis);
    std::shared_ptr<TOps::TOp<Cx, 6, 4>> loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(FT, nSlab);
    std::shared_ptr<TOps::TOp<Cx, 5, 6>> slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(FT->ishape, nSlab);
    std::shared_ptr<TOps::TOp<Cx, 5, 4>> compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
    return compose1;
  } else {
    if (gridOpts.vcc) {
      auto       nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, nC, basis, shape);
      auto const ns = nufft->ishape;
      auto       reshape =
        std::make_shared<TOps::ReshapeInput<TOps::NUFFT<3, true>, 5>>(nufft, Sz5{ns[0] * ns[1], ns[2], ns[3], ns[4], ns[5]});
      if (nSlab == 1) {
        auto rout =
          std::make_shared<TOps::ReshapeOutput<decltype(reshape)::element_type, 4>>(reshape, AddBack(reshape->oshape, 1));
        return rout;
      } else {
        auto loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(reshape, nSlab);
        auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(reshape->ishape, nSlab);
        auto compose2 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
        return compose2;
      }
    } else {
      auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, nC, basis, shape);
      if (nSlab == 1) {
        auto reshape = std::make_shared<TOps::ReshapeOutput<TOps::NUFFT<3, false>, 4>>(nufft, AddBack(nufft->oshape, 1));
        return reshape;
      } else {
        auto loop = std::make_shared<TOps::Loop<TOps::NUFFT<3, false>>>(nufft, nSlab);
        auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(nufft->ishape, nSlab);
        auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
        return compose1;
      }
    }
  }
}

} // namespace Recon
} // namespace rl