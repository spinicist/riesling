#include "recon.hpp"

#include "op/compose.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/ndft.hpp"
#include "op/nufft-decant.hpp"
#include "op/nufft-lowmem.hpp"
#include "op/nufft.hpp"
#include "op/reshape.hpp"
#include "op/sense.hpp"

namespace rl {
namespace Recon {

auto Choose(GridOpts &gridOpts, SENSE::Opts &senseOpts, Trajectory const &traj, Basis::CPtr b, Cx5 const &noncart)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  if (nC == 1) {
    return Recon::Single(gridOpts, traj, nS, nT, b);
  } else {
    auto const kernels = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (senseOpts.decant) {
      return Decant(gridOpts, traj, nS, nT, b, kernels, traj.matrixForFOV(gridOpts.fov.Get()));
    } else {
      return SENSE(gridOpts, traj, nS, nT, b, kernels);
    }
  }
}

auto Single(GridOpts &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (nSlab > 1) { throw Log::Failure("Recon", "Multislab and 1 channel not supported right now"); }
  auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, 1, b, traj.matrix());
  auto ri = TOps::MakeReshapeInput(nufft, LastN<4>(nufft->ishape));
  auto ro = TOps::MakeReshapeOutput(ri, AddBack(ri->oshape, 1));
  auto timeLoop = TOps::MakeLoop(ro, nTime);
  return timeLoop;
}

auto SENSE(GridOpts &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b, Cx5 const &skern)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (gridOpts.lowmem) {
    auto nufft = TOps::NUFFTLowmem<3>::Make(traj, traj.matrixForFOV(gridOpts.fov.Get()), gridOpts, skern, b);
    if (nSlab > 1) {
      throw Log::Failure("Recon", "Lowmem and multislab not supported yet");
    } else {
      auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
      return TOps::MakeLoop(rn, nTime);
    }
  } else {
    Cx5 const smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov.Get()), gridOpts.osamp.Get());
    if (gridOpts.vcc) {
      auto sense = std::make_shared<TOps::VCCSENSE>(smaps, b ? b->nB() : 1);
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
      auto sense = std::make_shared<TOps::SENSE>(smaps, b ? b->nB() : 1);
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
}

auto Decant(GridOpts         &gridOpts,
            Trajectory const &traj,
            Index const       nSlab,
            Index const       nTime,
            Basis::CPtr       b,
            Cx5 const        &skern,
            Sz3 const        &matrix) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (gridOpts.vcc) {
    throw Log::Failure("Decant", "Not yet");
  } else {
    auto nufft = TOps::NUFFTDecant<3>::Make(traj, gridOpts, skern, b, matrix);
    if (nSlab > 1) {
      throw Log::Failure("Decant", "Not yet");
    } else {
      auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
      return TOps::MakeLoop(rn, nTime);
    }
  }
  return nullptr;
}

} // namespace Recon
} // namespace rl