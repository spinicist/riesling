#include "sense.hpp"

#include "tensors.hpp"
#include "threads.hpp"

#include "log.hpp"

#include <numbers>

constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;

namespace rl::TOps {

SENSE::SENSE(Cx5 const &maps, Index const frames)
  : Parent("SENSEOp",
           AddFront(LastN<3>(maps.dimensions()), frames),
           AddFront(LastN<3>(maps.dimensions()), maps.dimension(0), frames))
  , maps_{std::move(maps)}
{
  if (!(maps_.dimension(1) == 1 || maps_.dimension(1) == frames)) {
    Log::Fail("SENSE maps had {} frames, expected {}", maps_.dimension(1), frames);
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

void SENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  finishForward(y, time, false);
}

void SENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = ConjugateSum(y, maps_.broadcast(brdMaps));
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
  x.device(Threads::GlobalDevice()) += ConjugateSum(y, maps_.broadcast(brdMaps));
  finishAdjoint(x, time, true);
}

auto SENSE::nChannels() const -> Index { return oshape[0]; }
auto SENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

NonSENSE::NonSENSE(Cx4 const &img, Index const nC)
  : Parent("NonSENSEOp", AddFront(img.dimensions(), nC), AddFront(img.dimensions(), nC))
  , img_{img}
{
  res_.set(1, img_.dimension(0));
  res_.set(2, img_.dimension(1));
  res_.set(3, img_.dimension(2));
  res_.set(4, img_.dimension(3));
  brd_.set(0, nC);
}

void NonSENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = img_.reshape(res_).broadcast(brd_) * x;
  finishForward(y, time, false);
}

void NonSENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = img_.reshape(res_).broadcast(brd_).conjugate() * y;
  finishAdjoint(x, time, false);
}

void NonSENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::GlobalDevice()) += img_.reshape(res_).broadcast(brd_) * x;
  finishForward(y, time, true);
}

void NonSENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::GlobalDevice()) += img_.reshape(res_).broadcast(brd_).conjugate() * y;
  finishAdjoint(x, time, true);
}

auto NonSENSE::nChannels() const -> Index { return oshape[0]; }
auto NonSENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

VCCSENSE::VCCSENSE(Cx5 const &maps, Index const frames)
  : Parent("VCCSENSE",
           AddFront(LastN<3>(maps.dimensions()), frames),
           AddFront(LastN<3>(maps.dimensions()), maps.dimension(0), 2, frames))
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

void VCCSENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  y.chip<1>(0).device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
  y.chip<1>(1).device(Threads::GlobalDevice()) =
    x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  finishForward(y, time, false);
}

void VCCSENSE::adjoint(OutCMap const &y, InMap &x) const
{
  Eigen::IndexList<Eigen::type2index<0>> zero;

  auto const time = startAdjoint(y, x, false);
  auto const real = (y.chip<1>(0) * maps_.broadcast(brdMaps).conjugate()).sum(zero);
  auto const virt = (y.chip<1>(1) * maps_.broadcast(brdMaps)).sum(zero);
  x.device(Threads::GlobalDevice()) = (real + virt) * Cx(inv_sqrt2);

  finishAdjoint(x, time, false);
}

void VCCSENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  y.chip<1>(0).device(Threads::GlobalDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
  y.chip<1>(1).device(Threads::GlobalDevice()) +=
    x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  finishForward(y, time, true);
}

void VCCSENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  Eigen::IndexList<Eigen::type2index<0>> zero;

  auto const time = startAdjoint(y, x, true);
  auto const real = (y.chip<1>(0) * maps_.broadcast(brdMaps).conjugate()).sum(zero);
  auto const virt = (y.chip<1>(1) * maps_.broadcast(brdMaps)).sum(zero);
  x.device(Threads::GlobalDevice()) += (real + virt) * Cx(inv_sqrt2);

  finishAdjoint(x, time, true);
}

auto VCCSENSE::nChannels() const -> Index { return oshape[0]; }
auto VCCSENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl::TOps
