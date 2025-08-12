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

template <int ND>
auto Single(GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, Index const nSlab, Index const nTime, Basis::CPtr b)
  -> TOps::TOp<5, 5>::Ptr
{
  auto nufft = TOps::MakeNUFFT<ND>(gridOpts, traj, 1, b);
  if constexpr (ND == 2) {
    auto                 ri = TOps::MakeReshapeInput(nufft, Concatenate(FirstN<2>(nufft->ishape), LastN<1>(nufft->ishape)));
    TOps::TOp<4, 4>::Ptr sliceLoop = TOps::MakeLoop<2, 3>(ri, nSlab);
    TOps::TOp<5, 5>::Ptr timeLoop = TOps::MakeLoop<4, 4>(sliceLoop, nTime);
    return timeLoop;
  } else {
    if (nSlab > 1) { throw Log::Failure("Recon", "Multislab and 1 channel not supported right now"); }
    auto ri = TOps::MakeReshapeInput(nufft, Concatenate(FirstN<3>(nufft->ishape), LastN<1>(nufft->ishape)));
    auto ro = TOps::MakeReshapeOutput(ri, AddBack(ri->oshape, 1));
    auto timeLoop = TOps::MakeLoop<4, 4>(ro, nTime);
    return timeLoop;
  }
}

auto LowmemSENSE(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr b, Cx5 const &skern)
  -> TOps::TOp<5, 5>::Ptr
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
  -> TOps::TOp<5, 5>::Ptr
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

template <int ND> Recon<ND>::Recon(ReconOpts const       &rOpts,
                                   PreconOpts const      &pOpts,
                                   GridOpts<ND> const    &gridOpts,
                                   SENSE::Opts<ND> const &senseOpts,
                                   TrajectoryN<ND> const &traj,
                                   Basis::CPtr            b,
                                   Cx5 const             &noncart)
{
  Index const nChan = noncart.dimension(0);
  Index const nSlab = noncart.dimension(3);
  Index const nTime = noncart.dimension(4);
  if (nChan == 1) {
    A = Single(gridOpts, traj, nSlab, nTime, b);
    M = MakeKSpacePrecon(pOpts, gridOpts, traj, 1, Sz2{nSlab, nTime});
  } else {
    auto const skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
    if (rOpts.decant) {
      if constexpr (ND == 2) {
        throw(Log::Failure("recon", "DECANTER in 2D makes no SENSE (👏)"));
      } else {
        A = Decant(gridOpts, traj, nSlab, nTime, b, skern);
        M = MakeKSpacePrecon(pOpts, gridOpts, traj, nChan, Sz2{nSlab, nTime});
      }
    } else if (rOpts.lowmem) {
      if constexpr (ND == 2) {
        throw(Log::Failure("recon", "Lowmem in 2D not supported yet"));
      } else {
        A = LowmemSENSE(gridOpts, traj, nSlab, nTime, b, skern);
        M = MakeKSpacePrecon(pOpts, gridOpts, traj, nChan, Sz2{nSlab, nTime});
      }
    } else {
      auto sense =
        TOps::MakeSENSE(SENSE::KernelsToMaps<ND>(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp), b ? b->nB() : 1);
      auto nufft = TOps::MakeNUFFT(gridOpts, traj, sense->nChannels(), b);
      if constexpr (ND == 2) {
        auto slices = TOps::MakeLoop<2, 3>(nufft, nSlab);
        auto ss = TOps::MakeCompose(sense, slices);
        auto time = TOps::MakeLoop<4, 4>(ss, nTime);
        A = time;
        M = MakeKSpacePrecon(pOpts, gridOpts, traj, nChan, Sz2{nSlab, nTime});
      } else {
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
          M = MakeKSpacePrecon(pOpts, gridOpts, traj, sense->maps(), Sz2{nSlab, nTime});
        }
      }
    }
  }
}

template <int ND> Recon<ND>::Recon(ReconOpts const       &rOpts,
                                   PreconOpts const      &pOpts,
                                   GridOpts<ND> const    &gridOpts,
                                   SENSE::Opts<ND> const &senseOpts,
                                   TrajectoryN<ND> const &traj,
                                   f0Opts const          &f0opts,
                                   Cx5 const             &noncart,
                                   Re3 const             &f0map)
{
  Index const nSamp = noncart.dimension(1);
  Index const nSlice = noncart.dimension(3);
  Index const nTime = noncart.dimension(4);
  auto const  skern = SENSE::Choose(senseOpts, gridOpts, traj, noncart);
  auto        F = std::make_shared<TOps::f0Segment>(f0map, f0opts.τacq, f0opts.Nτ, nSamp);
  auto        b = F->basis();
  auto        S = TOps::MakeSENSE(SENSE::KernelsToMaps<ND>(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp), b->nB());
  auto        SF = TOps::MakeCompose(F, S);
  auto        N = TOps::MakeNUFFT<ND>(gridOpts, traj, S->nChannels(), b);
  if constexpr (ND == 2) {
    auto NL = TOps::MakeLoop<2, 3>(N, nSlice);
    auto NLSF = TOps::MakeCompose(SF, NL);
    auto NLSFT = TOps::MakeLoop<4, 4>(NLSF, nTime);
    A = NLSFT;
    M = MakeKSpacePrecon(pOpts, gridOpts, traj, S->nChannels(), Sz2{nSlice, nTime});
  } else {
    if (nSlice > 1) {
      throw(Log::Failure("Recon", "Not supported right now"));
    } else {
      auto NSF = TOps::MakeCompose(SF, N);
      if (nTime > 1) {
        auto NSF2 = TOps::MakeReshapeOutput(NSF, AddBack(NSF->oshape, 1));
        A = TOps::MakeLoop<4, 4>(NSF2, nTime);
      } else {
        auto NSF2 = TOps::MakeReshapeOutput(NSF, AddBack(NSF->oshape, 1, 1));
        A = TOps::MakeReshapeInput(NSF2, AddBack(NSF2->ishape, 1));
      }
      M = MakeKSpacePrecon(pOpts, gridOpts, traj, S->maps(), Sz2{nSlice, nTime});
    }
  }
}

template struct Recon<2>;
template struct Recon<3>;

} // namespace rl