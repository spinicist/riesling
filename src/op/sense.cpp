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

void SenseOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x);
  y.device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps);
  finishForward(y, time);
}

void SenseOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y);
  x.device(Threads::GlobalDevice()) = ConjugateSum(y, maps_.reshape(resMaps).broadcast(brdMaps));
  finishAdjoint(x, time);
}

  auto SenseOp::nChannels() const -> Index { return oshape[0]; }
  auto SenseOp::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl
