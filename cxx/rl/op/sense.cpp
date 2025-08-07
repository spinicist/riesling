#include "sense.hpp"

#include "../log/log.hpp"
#include "../sense/sense.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

auto SENSEOp::Make(Cx5 const &maps, Index const nB) -> Ptr { return std::make_shared<SENSEOp>(maps, nB); }

SENSEOp::SENSEOp(Cx5 const &maps, Index const nB)
  : Parent("SENSEOp")
  , maps_{maps}
{
  for (int ii = 0; ii < 3; ii++) {
    ishape[ii] = maps_.dimension(ii);
    oshape[ii] = maps_.dimension(ii);
    resX.set(ii, maps_.dimension(ii));
  }
  ishape[3] = nB;
  oshape[3] = nB;
  resX.set(3, nB);
  oshape[4] = maps_.dimension(4);
  brdX.set(4, maps_.dimension(4));

  if (maps_.dimension(3) == 1) {
    brdMaps.set(3, nB);
  } else if (maps_.dimension(3) == nB) {
    brdMaps.set(3, 1);
  } else {
    throw Log::Failure("SENSEOp", "Maps had basis size {} expected {}", maps_.dimension(3), nB);
  }
}

void SENSEOp::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * y.constant(s);
  this->finishForward(y, time, false);
}

void SENSEOp::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * y.constant(s);
  this->finishForward(y, time, true);
}

void SENSEOp::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{4}) * x.constant(s);
  this->finishAdjoint(x, time, false);
}

void SENSEOp::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{4}) * x.constant(s);
  this->finishAdjoint(x, time, true);
}

auto SENSEOp::nChannels() const -> Index { return oshape[4]; }
auto SENSEOp::mapDimensions() const -> Sz3 { return FirstN<3>(ishape); }
auto SENSEOp::maps() const -> Cx5 { return maps_; }

auto MakeSENSE(Cx5 const &maps, Index const nB) -> SENSEOp::Ptr { return std::make_shared<SENSEOp>(maps, nB); }

} // namespace rl::TOps
