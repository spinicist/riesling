#include "sense.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

SENSE::SENSE(Cx5 const &maps, Index const nB)
  : Parent("SENSEOp",
           AddBack(FirstN<3>(maps.dimensions()), nB),
           AddBack(FirstN<3>(maps.dimensions()), maps.dimension(3), nB))
  , maps_{maps}
{
  resX.set(0, maps_.dimension(0));
  resX.set(1, maps_.dimension(1));
  resX.set(2, maps_.dimension(2));
  resX.set(4, nB);
  brdX.set(3, maps_.dimension(3));

  if (maps_.dimension(4) == 1) {
    brdMaps.set(4, nB);
  } else if (maps_.dimension(0) == nB) {
    brdMaps.set(4, 1);
  } else {
    throw Log::Failure("SENSE", "Maps had basis size {} expected {}", maps_.dimension(4), nB);
  }
}

void SENSE::forward(InCMap const x, OutMap y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time, false);
}

void SENSE::iforward(InCMap const x, OutMap y) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time, true);
}

void SENSE::adjoint(OutCMap const y, InMap x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  finishAdjoint(x, time, false);
}

void SENSE::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  finishAdjoint(x, time, true);
}

auto SENSE::nChannels() const -> Index { return oshape[3]; }
auto SENSE::mapDimensions() const -> Sz3 { return FirstN<3>(ishape); }

} // namespace rl::TOps
