#include "recon.hpp"

#include "compose.hpp"
#include "f0.hpp"
#include "loop.hpp"
#include "multiplex.hpp"
#include "ndft.hpp"
#include "nufft-decant.hpp"
#include "nufft-lowmem.hpp"
#include "nufft.hpp"
#include "reshape.hpp"
#include "sense.hpp"

namespace rl {

namespace {
auto Single(GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (nSlab > 1) { throw Log::Failure("Recon", "Multislab and 1 channel not supported right now"); }
  auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, 1, b);
  auto ri = TOps::MakeReshapeInput(nufft, LastN<4>(nufft->ishape));
  auto ro = TOps::MakeReshapeOutput(ri, AddBack(ri->oshape, 1));
  auto timeLoop = TOps::MakeLoop<4, 4>(ro, nTime);
  return timeLoop;
}

auto LowmemSENSE(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b, Cx5 const &skern)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto nufft = TOps::NUFFTLowmem<3>::Make(gridOpts, traj, skern, b);
  if (nSlab > 1) {
    throw Log::Failure("Recon", "Lowmem and multislab not supported yet");
  } else {
    auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
    return TOps::MakeLoop<4, 4>(rn, nTime);
  }
}

auto Decant(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b, Cx5 const &skern)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto nufft = TOps::NUFFTDecant<3>::Make(gridOpts, traj, skern, b);
  if (nSlab > 1) {
    throw Log::Failure("Decant", "Not yet");
  } else {
    auto rn = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
    return TOps::MakeLoop<4, 4>(rn, nTime);
  }
  return nullptr;
}
} // namespace

Recon::Recon(ReconOpts const      &rOpts,
             PreconOpts const     &pOpts,
             GridOpts<3> const    &gridOpts,
             SENSE::Opts<3> const &senseOpts,
             Trajectory const     &traj,
             Basis::CPtr           b,
             Cx5 const            &noncart)
{
  Index const nChan = noncart.dimension(0);
  Index const nSlab = noncart.dimension(3);
  Index const nTime = noncart.dimension(4);
  if (nChan == 1) {
    A = Single(gridOpts, traj, nSlab, nTime, b);
  } else {
    auto const skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (rOpts.decant) {
      A = Decant(gridOpts, traj, nSlab, nTime, b, skern);
      M = MakeKSpacePrecon(pOpts, gridOpts, traj, nChan, Sz2{nSlab, nTime});
    } else if (rOpts.lowmem) {
      A = LowmemSENSE(gridOpts, traj, nSlab, nTime, b, skern);
      M = MakeKSpacePrecon(pOpts, gridOpts, traj, nChan, Sz2{nSlab, nTime});
    } else {
      auto sense = TOps::MakeSENSE<3>(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp, b ? b->nB() : 1);
      auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, skern.dimension(3), b);
      if (nSlab > 1) {
        throw(Log::Failure("Recon", "Not supported right now"));
      } else {
        auto NS = TOps::MakeCompose(sense, nufft);
        if (nTime > 1) {
          auto NS2 = TOps::MakeReshapeOutput(NS, AddBack(NS->oshape, 1));
          A = TOps::MakeLoop<4, 4>(NS2, nTime);
        } else {
          auto NS2 = TOps::MakeReshapeOutput(NS, AddBack(NS->oshape, 1, 1));
          A = TOps::MakeReshapeInput(NS2, AddBack(NS2->ishape, 1));
        }
      }
      M = MakeKSpacePrecon(pOpts, gridOpts, traj, sense->maps(), nSlab, nTime); // In case the SENSE op does move
    }
  }
}

Recon::Recon(ReconOpts const      &rOpts,
             PreconOpts const     &pOpts,
             GridOpts<3> const    &gridOpts,
             SENSE::Opts<3> const &senseOpts,
             Trajectory const     &traj,
             f0Opts const         &f0opts,
             Cx5 const            &noncart,
             Re3 const            &f0map)
{
  Index const nSamp = noncart.dimension(1);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);
  auto const  skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
  Cx5 const   smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp);
  M = MakeKSpacePrecon(pOpts, gridOpts, traj, smaps, nS, nT);

  auto f0 = std::make_shared<TOps::f0Segment>(f0map, f0opts.τacq, f0opts.Nτ, nSamp);
  auto b = f0->basis();
  auto sense = TOps::MakeSENSE<3>(smaps, b->nB());
  auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, smaps.dimension(3), b);
  auto slabLoop = TOps::MakeLoop<3, 3>(nufft, nS);
  if (nS > 1) {
    throw(Log::Failure("Recon", "Not supported right now"));
  } else {
    auto reshape = TOps::MakeReshapeOutput(TOps::MakeCompose(f0, sense), AddBack(sense->oshape, 1));
    A = TOps::MakeLoop<4, 4>(TOps::MakeCompose(reshape, slabLoop), nT);
  }
}

} // namespace rl