#include "sense.hpp"

#include "tensorOps.hpp"

namespace rl {

SenseOp::SenseOp(Cx4 const &maps, Index const d0)
  : maps_{std::move(maps)}
  , inSz_{d0, maps_.dimension(1), maps_.dimension(2), maps_.dimension(3)}
  , outSz_{maps_.dimension(0), d0, maps_.dimension(1), maps_.dimension(2), maps_.dimension(3)}
{
  resX.set(1, d0);
  resX.set(2, maps_.dimension(1));
  resX.set(3, maps_.dimension(2));
  resX.set(4, maps_.dimension(3));
  brdX.set(0, maps_.dimension(0));

  resMaps.set(0, maps_.dimension(0));
  resMaps.set(2, maps_.dimension(1));
  resMaps.set(3, maps_.dimension(2));
  resMaps.set(4, maps_.dimension(3));
  brdMaps.set(1, d0);
}

auto SenseOp::inputDimensions() const -> InputDims
{
  return inSz_;
}

auto SenseOp::outputDimensions() const -> OutputDims
{
  return outSz_;
}

auto SenseOp::forward(Input const &x) const -> Output
{
  return (x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps));
}

auto SenseOp::adjoint(Output const &x) const -> Input
{
  return ConjugateSum(x, maps_.reshape(resMaps).broadcast(brdMaps));
}

} // namespace rl
