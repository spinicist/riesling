#include "recon.hpp"

#include "op/compose.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

namespace rl {

auto make_recon(CoreOpts                       &coreOpts,
                SDC::Opts                      &sdcOpts,
                Trajectory const               &traj,
                std::shared_ptr<SenseOp> const &sense,
                Basis<Cx> const                &basis) -> std::shared_ptr<ReconOp>
{
  Index const nC = sense->nChannels();
  auto const  shape = sense->mapDimensions();
  auto const sdc = SDC::Choose(sdcOpts, nC, traj, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  if (coreOpts.ndft) {
    auto ndft = make_ndft(traj.points(), nC, shape, basis, sdc);
    return std::make_shared<ReconOp>(sense, ndft);
  } else {
    auto nufft = make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC, shape, basis, sdc, coreOpts.bucketSize.Get(),
                            coreOpts.splitSize.Get());
    return std::make_shared<ReconOp>(sense, nufft);
  }
}

} // namespace rl