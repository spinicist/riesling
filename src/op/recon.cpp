#include "recon.hpp"

#include "multiply.hpp"
#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

using ReconOp = MultiplyOp<SenseOp, Operator<Cx, 5, 4>>;

ReconOp Recon(
  CoreOpts &coreOpts,
  SDC::Opts &sdcOpts,
  SENSE::Opts &senseOpts,
  Trajectory const &traj,
  bool const toeplitz,
  HD5::Reader &reader)
{
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto sense = std::make_unique<SenseOp>(
    SENSE::Choose(senseOpts, coreOpts, sdcOpts, traj, reader), basis ? basis.value().dimension(1) : traj.nFrames());
  auto const sdc = SDC::Choose(sdcOpts, traj, sense->nChannels(), coreOpts.ktype.Get(), coreOpts.osamp.Get());
  auto nufft = make_nufft(
    traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), sense->nChannels(), sense->mapDimensions(), sdc, basis, toeplitz);
  MultiplyOp<SenseOp, Operator<Cx, 5, 4>> recon("ReconOp", sense, nufft);
  return recon;
}

} // namespace rl