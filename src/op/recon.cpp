#include "recon.hpp"

#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

namespace rl {

auto make_recon(
  CoreOpts &coreOpts, SDC::Opts &sdcOpts, Trajectory const &traj, std::shared_ptr<SenseOp> const &sense, Re2 const &basis)
  -> std::shared_ptr<ReconOp>
{

  auto const sdc = SDC::Choose(sdcOpts, sense->nChannels(), traj, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  auto       nufft =
    make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), sense->nChannels(), sense->mapDimensions(), basis, sdc);
  return std::make_shared<ReconOp>(sense, nufft);
}

} // namespace rl