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

auto SENSE(bool const        ndft,
           GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Index const       nTime,
           Basis const      &basis,
           Cx5 const        &data) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (ndft) {
    if (gridOpts.vcc) { Log::Warn("VCC and NDFT not supported yet"); }
    auto sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
    auto nufft = TOps::NDFT<3>::Make(sense->mapDimensions(), traj.points(), sense->nChannels(), basis);
    auto loop = TOps::MakeLoop(nufft, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto compose2 = TOps::MakeCompose(sense, compose1);
    auto timeLoop = TOps::MakeLoop(compose2, nTime);
    return timeLoop;
  } else {
    if (gridOpts.vcc) {
      auto sense = std::make_shared<TOps::VCCSENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
      auto nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, sense->nChannels(), basis, sense->mapDimensions());
      auto loop = TOps::MakeLoop(nufft, nSlab);
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 6>>(sense->oshape, nSlab);
      auto compose1 = TOps::MakeCompose(slabToVol, loop);
      auto compose2 = TOps::MakeCompose(sense, compose1);
      auto timeLoop = TOps::MakeLoop(compose2, nTime);
      return timeLoop;
    } else {
      auto sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), basis.dimension(0));
      auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, sense->nChannels(), basis, sense->mapDimensions());
      auto slabLoop = TOps::MakeLoop(nufft, nSlab);
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
      auto compose1 = TOps::MakeCompose(slabToVol, slabLoop);
      auto compose2 = TOps::MakeCompose(sense, compose1);
      auto timeLoop = TOps::MakeLoop(compose2, nTime);
      return timeLoop;
    }
  }
}

auto Channels(bool const        ndft,
              GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Index const       nTime,
              Basis const      &basis,
              Sz3 const         shape) -> TOps::TOp<Cx, 6, 5>::Ptr
{
  if (ndft) {
    auto FT = TOps::NDFT<3>::Make(shape, traj.points(), nC, basis);
    auto loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(FT, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(FT->ishape, nSlab);
    auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(compose1, nTime);
    return timeLoop;
  } else {
    if (gridOpts.vcc) {
      auto       nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, nC, basis, shape);
      auto const ns = nufft->ishape;
      auto       reshape =
        std::make_shared<TOps::ReshapeInput<TOps::NUFFT<3, true>, 5>>(nufft, Sz5{ns[0] * ns[1], ns[2], ns[3], ns[4], ns[5]});
      if (nSlab == 1) {
        auto rout =
          std::make_shared<TOps::ReshapeOutput<decltype(reshape)::element_type, 4>>(reshape, AddBack(reshape->oshape, 1));
        auto timeLoop = TOps::MakeLoop(rout, nTime);
        return timeLoop;
      } else {
        auto loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(reshape, nSlab);
        auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(reshape->ishape, nSlab);
        auto compose2 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
        auto timeLoop = TOps::MakeLoop(compose2, nTime);
        return timeLoop;
      }
    } else {
      auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, nC, basis, shape);
      if (nSlab == 1) {
        auto reshape = std::make_shared<TOps::ReshapeOutput<TOps::NUFFT<3, false>, 4>>(nufft, AddBack(nufft->oshape, 1));
        auto timeLoop = TOps::MakeLoop(reshape, nTime);
        return timeLoop;
      } else {
        auto loop = std::make_shared<TOps::Loop<TOps::NUFFT<3, false>>>(nufft, nSlab);
        auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(nufft->ishape, nSlab);
        auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
        auto timeLoop = TOps::MakeLoop(compose1, nTime);
        return timeLoop;
      }
    }
  }
}

} // namespace Recon
} // namespace rl