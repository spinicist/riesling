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

auto Single(TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (nSlab > 1) { throw Log::Failure("Recon", "Multislab and 1 channel not supported right now"); }
  auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, 1, b);
  auto ri = TOps::MakeReshapeInput(nufft, LastN<4>(nufft->ishape));
  auto ro = TOps::MakeReshapeOutput(ri, AddBack(ri->oshape, 1));
  auto timeLoop = TOps::MakeLoop(ro, nTime);
  return timeLoop;
}

auto LowmemSENSE(TOps::Grid<3>::Opts const &gridOpts,
                 Trajectory const          &traj,
                 Index const                nSlab,
                 Index const                nTime,
                 Basis::CPtr                b,
                 Cx5 const                 &skern) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto nufft = TOps::NUFFTLowmem<3>::Make(gridOpts, traj, skern, b);
  if (nSlab > 1) {
    throw Log::Failure("Recon", "Lowmem and multislab not supported yet");
  } else {
    auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
    return TOps::MakeLoop(rn, nTime);
  }
}

auto SENSERecon(TOps::Grid<3>::Opts const &gridOpts,
                Trajectory const          &traj,
                Index const                nSlab,
                Index const                nTime,
                Basis::CPtr                b,
                Cx5 const                 &skern) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  Cx5 const smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp);
  auto      sense = std::make_shared<TOps::SENSE>(smaps, gridOpts.vcc, b ? b->nB() : 1);
  auto      nufft = TOps::NUFFT<3>::Make(gridOpts, traj, smaps.dimension(1), b);
  auto      slabLoop = TOps::MakeLoop(nufft, nSlab);
  if (nSlab > 1) {
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
    return TOps::MakeLoop(TOps::MakeCompose(sense, TOps::MakeCompose(slabToVol, slabLoop)), nTime);
  } else {
    auto reshape = TOps::MakeReshapeOutput(sense, AddBack(sense->oshape, 1));
    return TOps::MakeLoop(TOps::MakeCompose(reshape, slabLoop), nTime);
  }
}

auto Decant(TOps::Grid<3>::Opts const &gridOpts,
            Trajectory const          &traj,
            Index const                nSlab,
            Index const                nTime,
            Basis::CPtr                b,
            Cx5 const                 &skern) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto nufft = TOps::NUFFTDecant<3>::Make(gridOpts, traj, skern, b);
  if (nSlab > 1) {
    throw Log::Failure("Decant", "Not yet");
  } else {
    auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
    return TOps::MakeLoop(rn, nTime);
  }
  return nullptr;
}

Recon::Recon(Opts const                &rOpts,
             PreconOpts const          &pOpts,
             TOps::Grid<3>::Opts const &gridOpts,
             SENSE::Opts               &senseOpts,

             Trajectory const &traj,
             Basis::CPtr       b,
             Cx5 const        &noncart)
{
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  if (nC == 1) {
    A = Single(gridOpts, traj, nS, nT, b);
  } else {
    auto const kernels = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (rOpts.decant) {
      A = Decant(gridOpts, traj, nS, nT, b, kernels);
    } else if (rOpts.lowmem) {
      A = LowmemSENSE(gridOpts, traj, nS, nT, b, kernels);
    } else {
      A = SENSERecon(gridOpts, traj, nS, nT, b, kernels);
    }
  }
  M = MakeKSpaceSingle(pOpts, gridOpts, traj, nC, nS, nT, b);
}

} // namespace rl