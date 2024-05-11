#include "recon.hpp"

#include "op/compose.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"

namespace rl {

auto SENSERecon(CoreOpts                       &coreOpts,
                GridOpts                       &gridOpts,
                Trajectory const               &traj,
                Index const                     nSlab,
                std::shared_ptr<SenseOp> const &sense,
                Basis<Cx> const                &basis) -> TensorOperator<Cx, 4, 4>::Ptr
{
  Index const nC = sense->nChannels();
  auto const  shape = sense->mapDimensions();

  std::shared_ptr<TensorOperator<Cx, 5, 3>> FT = nullptr;
  if (coreOpts.ndft) {
    FT = NDFTOp<3>::Make(shape, traj.points(), nC, basis);
  } else {
    FT = NUFFTOp<3>::Make(shape, traj, gridOpts, nC, basis);
  }

  std::shared_ptr<TensorOperator<Cx, 6, 4>> loop = std::make_shared<LoopOp<TensorOperator<Cx, 5, 3>>>(FT, nSlab);
  std::shared_ptr<TensorOperator<Cx, 5, 6>> slabToVol = std::make_shared<Multiplex<Cx, 5>>(sense->oshape, nSlab);

  std::shared_ptr<TensorOperator<Cx, 5, 4>> compose1 = std::make_shared<decltype(Compose(slabToVol, loop))>(slabToVol, loop);
  std::shared_ptr<TensorOperator<Cx, 4, 4>> compose2 = std::make_shared<decltype(Compose(sense, compose1))>(sense, compose1);
  return compose2;
}

auto Channels(CoreOpts         &coreOpts,
              GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Basis<Cx> const  &basis) -> TensorOperator<Cx, 5, 4>::Ptr
{
  Index const nB = basis.dimension(0);
  auto const  shape = traj.matrixForFOV(coreOpts.fov.Get());

  std::shared_ptr<TensorOperator<Cx, 5, 3>> FT = nullptr;
  if (coreOpts.ndft) {
    FT = NDFTOp<3>::Make(shape, traj.points(), nC, basis);
  } else {
    FT = NUFFTOp<3>::Make(shape, traj, gridOpts, nC, basis);
  }

  std::shared_ptr<TensorOperator<Cx, 6, 4>> loop = std::make_shared<LoopOp<TensorOperator<Cx, 5, 3>>>(FT, nSlab);
  std::shared_ptr<TensorOperator<Cx, 5, 6>> slabToVol = std::make_shared<Multiplex<Cx, 5>>(AddFront(shape, nC, nB), nSlab);

  std::shared_ptr<TensorOperator<Cx, 5, 4>> compose1 = std::make_shared<decltype(Compose(slabToVol, loop))>(slabToVol, loop);
  return compose1;
}

} // namespace rl