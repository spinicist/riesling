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
                Cx5 const                 &smaps) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  auto sense = std::make_shared<TOps::SENSE>(smaps, gridOpts.vcc, b ? b->nB() : 1);
  auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, smaps.dimension(1), b);
  auto slabLoop = TOps::MakeLoop(nufft, nSlab);
  if (nSlab > 1) {
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
    return TOps::MakeLoop(TOps::MakeCompose(sense, TOps::MakeCompose(slabToVol, slabLoop)), nTime);
  } else {
    auto reshape = TOps::MakeReshapeOutput(sense, AddBack(sense->oshape, 1));
    return TOps::MakeLoop(TOps::MakeCompose(reshape, slabLoop), nTime);
  }
}

auto SENSEf0Recon(TOps::Grid<3>::Opts const &gridOpts,
                  Trajectory const          &traj,
                  Index const                nSlab,
                  Index const                nTime,
                  Basis::CPtr                b,
                  Recon::f0Opts const       &f0Opts,
                  Cx5 const                 &smaps) -> TOps::TOp<Cx, 5, 5>::Ptr
{
  if (!b) { throw Log::Failure("f0Recon", "A basis is required for f0 correction"); }
  if (b->nB() != f0Opts.τ.size()) {
    throw Log::Failure("f0Recon", "Basis size {} did not match τ {}", b->nB(), f0Opts.τ.size());
  }

  HD5::Reader f0_file(f0Opts.fname);
  auto const  f0_map = f0_file.readTensor<Re3>("f0");
  auto        f0 = std::make_shared<TOps::f0Segment>(f0_map, f0Opts.τ);
  auto        sense = std::make_shared<TOps::SENSE>(smaps, gridOpts.vcc, b->nB());
  auto        nufft = TOps::NUFFT<3>::Make(gridOpts, traj, smaps.dimension(1), b);
  auto        slabLoop = TOps::MakeLoop(nufft, nSlab);
  if (nSlab > 1) {
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(sense->oshape, nSlab);
    return TOps::MakeLoop(TOps::MakeCompose(TOps::MakeCompose(f0, sense), TOps::MakeCompose(slabToVol, slabLoop)), nTime);
  } else {
    auto reshape = TOps::MakeReshapeOutput(TOps::MakeCompose(f0, sense), AddBack(sense->oshape, 1));
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
             SENSE::Opts const         &senseOpts,
             Trajectory const          &traj,
             Basis::CPtr                b,
             f0Opts const              &f0Opts,
             Cx5 const                 &noncart)
{
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  if (nC == 1) {
    A = Single(gridOpts, traj, nS, nT, b);
  } else {
    auto const skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (rOpts.decant) {
      A = Decant(gridOpts, traj, nS, nT, b, skern);
      M = MakeKSpaceSingle(pOpts, gridOpts, traj, nC, nS, nT);
    } else if (rOpts.lowmem) {
      A = LowmemSENSE(gridOpts, traj, nS, nT, b, skern);
      M = MakeKSpaceSingle(pOpts, gridOpts, traj, nC, nS, nT);
    } else {
      Cx5 const smaps = SENSE::KernelsToMaps(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp);
      M = MakeKSpaceMulti(pOpts, gridOpts, traj, smaps, nS, nT); // In case the SENSE op does move
      if (f0Opts.fname.size()) {
        A = SENSEf0Recon(gridOpts, traj, nS, nT, b, f0Opts, smaps);
      } else {
        A = SENSERecon(gridOpts, traj, nS, nT, b, smaps);
      }
    }
  }
}

} // namespace rl