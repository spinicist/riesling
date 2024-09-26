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

auto Single(GridOpts &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (nSlab > 1) { throw Log::Failure("Recon", "Multislab and 1 channel not supported right now"); }
  auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, 1, b);
  auto ri = TOps::MakeReshapeInput(nufft, LastN<4>(nufft->ishape));
  auto ro = TOps::MakeReshapeOutput(ri, AddBack(ri->oshape, 1));
  auto timeLoop = TOps::MakeLoop(ro, nTime);
  return timeLoop;
}

auto SENSE(GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Index const       nTime,
           Basis::CPtr       b,
           Cx5 const        &data) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (gridOpts.vcc) {
    auto sense = std::make_shared<TOps::VCCSENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), b ? b->nB() : 1);
    auto nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, sense->nChannels(), b, sense->mapDimensions());
    auto slabLoop = TOps::MakeLoop(nufft, nSlab);
    if (nSlab > 1) {
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 6>>(sense->oshape, nSlab);
      return TOps::MakeLoop(TOps::MakeCompose(sense, TOps::MakeCompose(slabToVol, slabLoop)), nTime);
    } else {
      auto reshape = TOps::MakeReshapeOutput(sense, AddBack(sense->oshape, 1));
      return TOps::MakeLoop(TOps::MakeCompose(reshape, slabLoop), nTime);
    }
  } else {
    auto sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, data), b ? b->nB() : 1);
    auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, sense->nChannels(), b, sense->mapDimensions());
    auto slabLoop = TOps::MakeLoop(nufft, nSlab);
    if (nSlab > 1) {
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
      return TOps::MakeLoop(TOps::MakeCompose(sense, TOps::MakeCompose(slabToVol, slabLoop)), nTime);
    } else {
      auto reshape = TOps::MakeReshapeOutput(sense, AddBack(sense->oshape, 1));
      return TOps::MakeLoop(TOps::MakeCompose(reshape, slabLoop), nTime);
    }
  }
}

auto Channels(GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Index const       nTime,
              Basis::CPtr       basis,
              Sz3 const         shape) -> TOps::TOp<Cx, 6, 5>::Ptr
{
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

} // namespace Recon
} // namespace rl