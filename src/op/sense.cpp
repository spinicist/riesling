#include "sense.hpp"

#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

SenseOp::SenseOp(Cx5 const &maps, Index const frames)
  : Parent("SENSEOp", AddFront(LastN<3>(maps.dimensions()), frames), AddFront(LastN<3>(maps.dimensions()), maps.dimension(0), frames))
  , maps_{std::move(maps)}
{
  if (!(maps.dimension(1) == 1 || maps.dimension(1) == frames)) {
    Log::Fail("SENSE maps had {} frames, expected {}", maps.dimension(1), frames);
  }

  resX.set(1, frames);
  resX.set(2, maps_.dimension(2));
  resX.set(3, maps_.dimension(3));
  resX.set(4, maps_.dimension(4));
  brdX.set(0, maps_.dimension(0));

  if (maps.dimension(1) == 1) {
    brdMaps.set(1, frames);
  } else {
    brdMaps.set(1, 1);
  }
}

void SenseOp::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x);
  y.device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time);
}

void SenseOp::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y);
  x.device(Threads::GlobalDevice()) = ConjugateSum(y, maps_.broadcast(brdMaps));
  finishAdjoint(x, time);
}

auto SenseOp::nChannels() const -> Index { return oshape[0]; }
auto SenseOp::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl
