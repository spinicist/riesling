#include "sense.hpp"

#include "tensors.hpp"
#include "threads.hpp"

#include "log.hpp"

#include <numbers>

constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;

namespace rl::TOps {

SENSE::SENSE(Cx5 const &maps, Index const nB)
  : Parent("SENSEOp",
           AddFront(LastN<3>(maps.dimensions()), nB),
           AddFront(LastN<3>(maps.dimensions()), nB, maps.dimension(1)))
  , maps_{std::move(maps)}
{
  resX.set(0, nB);
  resX.set(2, maps_.dimension(2));
  resX.set(3, maps_.dimension(3));
  resX.set(4, maps_.dimension(4));
  brdX.set(1, maps_.dimension(1));

  if (maps_.dimension(0) == 1) {
    brdMaps.set(0,nB);
  } else if (maps_.dimension(0) == nB) {
    brdMaps.set(0, 1);
  } else {
    Log::Fail("SENSE maps had basis size {}, expected {}", maps_.dimension(0), nB);
  }
}

void SENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time, false);
}

void SENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = DimDot<1>(y, maps_.broadcast(brdMaps));
  finishAdjoint(x, time, false);
}

void SENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::GlobalDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time, true);
}

void SENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::GlobalDevice()) += DimDot<1>(y, maps_.broadcast(brdMaps));
  finishAdjoint(x, time, true);
}

auto SENSE::nChannels() const -> Index { return oshape[1]; }
auto SENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

EstimateKernels::EstimateKernels(Cx4 const &img, Index const nC)
  : Parent("EstimateKernelsOp",
           AddFront(LastN<3>(img.dimensions()), img.dimension(0), nC),
           AddFront(LastN<3>(img.dimensions()), img.dimension(0), nC))
  , img_{img}
{
  res_.set(0, img_.dimension(0));
  res_.set(2, img_.dimension(1));
  res_.set(3, img_.dimension(2));
  res_.set(4, img_.dimension(3));
  brd_.set(1, nC);
}

void EstimateKernels::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = img_.reshape(res_).broadcast(brd_) * x;
  finishForward(y, time, false);
}

void EstimateKernels::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = img_.reshape(res_).broadcast(brd_).conjugate() * y;
  finishAdjoint(x, time, false);
}

void EstimateKernels::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::GlobalDevice()) += img_.reshape(res_).broadcast(brd_) * x;
  finishForward(y, time, true);
}

void EstimateKernels::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::GlobalDevice()) += img_.reshape(res_).broadcast(brd_).conjugate() * y;
  finishAdjoint(x, time, true);
}

auto EstimateKernels::nChannels() const -> Index { return oshape[1]; }
auto EstimateKernels::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

VCCSENSE::VCCSENSE(Cx5 const &maps, Index const nB)
  : Parent("VCCSENSE",
           AddFront(LastN<3>(maps.dimensions()), nB),
           AddFront(LastN<3>(maps.dimensions()), nB, maps.dimension(1), 2))
  , maps_{std::move(maps)}
{
  resX.set(0, nB);
  resX.set(2, maps_.dimension(2));
  resX.set(3, maps_.dimension(3));
  resX.set(4, maps_.dimension(4));
  brdX.set(1, maps_.dimension(1));

  if (maps_.dimension(0) == 1) {
    brdMaps.set(0, nB);
  } else if (maps_.dimension(0) == nB) {
    brdMaps.set(0, 1);
  } else {
    Log::Fail("SENSE maps had basis size {}, expected {}", maps_.dimension(0), nB);
  }
}

void VCCSENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.chip<2>(0).device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
  y.chip<2>(1).device(Threads::GlobalDevice()) =
    x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  finishForward(y, time, false);
}

void VCCSENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  auto const real = DimDot<1>(y.chip<2>(0), maps_.broadcast(brdMaps));
  auto const virt = Sum<1>(y.chip<2>(1), maps_.broadcast(brdMaps));
  x.device(Threads::GlobalDevice()) = (real + virt) * Cx(inv_sqrt2);

  finishAdjoint(x, time, false);
}

void VCCSENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  y.chip<2>(0).device(Threads::GlobalDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
  y.chip<2>(1).device(Threads::GlobalDevice()) +=
    x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  finishForward(y, time, true);
}

void VCCSENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  auto const real = DimDot<1>(y.chip<2>(0), maps_.broadcast(brdMaps));
  auto const virt = Sum<1>(y.chip<2>(1), maps_.broadcast(brdMaps));
  x.device(Threads::GlobalDevice()) += (real + virt) * Cx(inv_sqrt2);

  finishAdjoint(x, time, true);
}

auto VCCSENSE::nChannels() const -> Index { return oshape[1]; }
auto VCCSENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl::TOps
