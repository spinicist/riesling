#include "recon2.hpp"

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
#include "shuffle.hpp"

namespace rl {

namespace {
auto Single(GridOpts<2> const &gridOpts, TrajectoryN<2> const &traj, Index const nSlice, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto nufft = TOps::NUFFT<2>::Make(gridOpts, traj, 1, b);
  auto reshape =
    TOps::MakeReshapeInput(nufft, AddBack(FirstN<2>(nufft->ishape), nufft->ishape[3])); // Strip the channel dimension
  auto slabLoop = TOps::MakeLoop(reshape, nSlice);
  auto shuff = TOps::MakeShuffleInput(slabLoop, Sz4{0, 1, 3, 2});
  auto timeLoop = TOps::MakeLoop(slabLoop, nTime);
  return timeLoop;
}

auto SENSERecon(GridOpts<2> const    &gridOpts,
                TrajectoryN<2> const &traj,
                Index const           nSlice,
                Index const           nTime,
                Basis::CPtr           b,
                Cx5 const            &smaps) -> TOps::TOp<Cx, 5, 5>::Ptr
{

  auto nufft = TOps::NUFFT<2>::Make(gridOpts, traj, smaps.dimension(3), b);
  auto slabLoop = TOps::MakeLoop(nufft, nSlice);
  auto shuff = TOps::MakeShuffleInput(slabLoop, Sz5{0, 1, 3, 4, 2});
  auto sense = std::make_shared<TOps::SENSE>(smaps, b ? b->nB() : 1);
  auto sc = TOps::MakeCompose(sense, shuff);
  TOps::TOp<Cx, 5, 5>::Ptr timeLoop = TOps::MakeLoop(sc, nTime);
  return timeLoop;
}

} // namespace

Recon2::Recon2(ReconOpts const      &rOpts,
               PreconOpts const     &pOpts,
               GridOpts<2> const    &gridOpts,
               SENSE::Opts const    &senseOpts,
               TrajectoryN<2> const &traj,
               Basis::CPtr           b,
               Cx5 const            &noncart)
{
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);
  if (nC == 1) {
    A = Single(gridOpts, traj, nS, nT, b);
  } else {
    auto const skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (rOpts.decant) {
      throw(Log::Failure("Recon2", "DECANT not implemented"));
    } else if (rOpts.lowmem) {
      throw(Log::Failure("Recon2", "Lowmem not implemented"));
    } else {
      Cx5 const smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp);
      M = MakeKSpacePrecon(pOpts, gridOpts, traj, smaps, nS, nT); // In case the SENSE op does move
      A = SENSERecon(gridOpts, traj, nS, nT, b, smaps);
    }
  }
}

// Recon2::Recon2(ReconOpts const      &rOpts,
//                PreconOpts const     &pOpts,
//                GridOpts<2> const    &gridOpts,
//                SENSE::Opts const    &senseOpts,
//                TrajectoryN<2> const &traj,
//                f0Opts const         &f0opts,
//                Cx5 const            &noncart,
//                Re3 const            &f0map)
// {
//   Index const nSamp = noncart.dimension(1);
//   Index const nS = noncart.dimension(3);
//   Index const nT = noncart.dimension(4);
//   auto const  skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
//   Cx5 const   smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp);
//   M = MakeKSpacePrecon(pOpts, gridOpts, traj, smaps, nS, nT); // In case the SENSE op does move

//   auto f0 = std::make_shared<TOps::f0Segment>(f0map, f0opts.τacq, f0opts.Nτ, nSamp);
//   auto b = f0->basis();
//   auto sense = std::make_shared<TOps::SENSE>(smaps, b->nB());
//   auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, smaps.dimension(3), b);
//   auto slabLoop = TOps::MakeLoop(nufft, nS);
//   if (nS > 1) {
//     auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nS);
//     A = TOps::MakeLoop(TOps::MakeCompose(TOps::MakeCompose(f0, sense), TOps::MakeCompose(slabToVol, slabLoop)), nT);
//   } else {
//     auto reshape = TOps::MakeReshapeOutput(TOps::MakeCompose(f0, sense), AddBack(sense->oshape, 1));
//     A = TOps::MakeLoop(TOps::MakeCompose(reshape, slabLoop), nT);
//   }
// }

} // namespace rl