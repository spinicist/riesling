#include "recon.hpp"

#include "op/compose.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"

namespace rl {

auto make_recon(CoreOpts                       &coreOpts,
                GridOpts                       &gridOpts,
                Trajectory const               &traj,
                std::shared_ptr<SenseOp> const &sense,
                Basis<Cx> const                &basis) -> std::shared_ptr<ReconOp>
{
  Index const nC = sense->nChannels();
  auto const  shape = sense->mapDimensions();
  if (coreOpts.ndft) {
    auto ndft = make_ndft(traj.points(), nC, shape, basis);
    return std::make_shared<ReconOp>(sense, ndft);
  } else {
    auto nufft = make_nufft(traj, gridOpts, nC, shape, basis);
    return std::make_shared<ReconOp>(sense, nufft);
  }
}

} // namespace rl