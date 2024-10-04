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

} // namespace Recon
} // namespace rl