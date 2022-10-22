#include "sense.hpp"

#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

SenseOp::SenseOp(Cx4 const &maps, Index const d0)
  : Parent("SENSEOp", AddFront(LastN<3>(maps.dimensions()), d0), AddFront(LastN<3>(maps.dimensions()), maps.dimension(0), d0))
  , maps_{std::move(maps)}
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

auto SenseOp::forward(InputMap x) const -> OutputMap
{
  auto const time = startForward(x);
  output().device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps);
  finishForward(output(), time);
  return output();
}

auto SenseOp::adjoint(OutputMap y) const -> InputMap
{
  auto const time = startAdjoint(y);
  input().device(Threads::GlobalDevice()) = ConjugateSum(y, maps_.reshape(resMaps).broadcast(brdMaps));
  finishAdjoint(input(), time);
  return input();
}

  auto SenseOp::nChannels() const -> Index { return outputDimensions()[0]; }
  auto SenseOp::mapDimensions() const -> Sz3 { return LastN<3>(inputDimensions()); }

} // namespace rl
