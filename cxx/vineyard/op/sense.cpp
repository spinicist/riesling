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
  Log::Tensor("nsx", x.dimensions(), x.data(), HD5::Dims::SENSE);
  Log::Tensor("nsy", y.dimensions(), y.data(), HD5::Dims::SENSE);
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

  resX.set(2, frames);
  resX.set(3, maps_.dimension(2));
  resX.set(4, maps_.dimension(3));
  resX.set(5, maps_.dimension(4));
  brdX.set(0, maps_.dimension(0));

  resMaps.set(0, maps_.dimension(0));
  resMaps.set(2, maps_.dimension(1));
  resMaps.set(3, maps_.dimension(2));
  resMaps.set(4, maps_.dimension(3));
  resMaps.set(5, maps_.dimension(4));

  if (maps.dimension(1) == 1) {
    brdMaps.set(2, frames);
  } else {
    brdMaps.set(2, 1);
  }
}

void VCCSENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  Sz6        st0{0, 0, 0, 0, 0, 0}, st1{0, 1, 0, 0, 0, 0}, sz = oshape;
  sz[1] = 1;
  y.slice(st0, sz).device(Threads::GlobalDevice()) =
    x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps) * maps_.constant(inv_sqrt2);
  y.slice(st1, sz).device(Threads::GlobalDevice()) =
    x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps).conjugate() * maps_.constant(inv_sqrt2);
  finishForward(y, time, false);
}

void VCCSENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  Sz6        st0{0, 0, 0, 0, 0, 0}, st1{0, 1, 0, 0, 0, 0}, sz = oshape;
  sz[1] = 1;
  x.device(Threads::GlobalDevice()) = (ConjugateSum(y.slice(st0, sz), maps_.reshape(resMaps).broadcast(brdMaps)) +
                                       ConjugateSum(y.slice(st1, sz), maps_.reshape(resMaps).broadcast(brdMaps).conjugate())) *
                                      maps_.constant(inv_sqrt2);
  finishAdjoint(x, time, false);
}

void VCCSENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  Sz6        st0{0, 0, 0, 0, 0, 0}, st1{0, 1, 0, 0, 0, 0}, sz = oshape;
  sz[1] = 1;
  y.slice(st0, sz).device(Threads::GlobalDevice()) +=
    x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps) * maps_.constant(inv_sqrt2);
  y.slice(st1, sz).device(Threads::GlobalDevice()) +=
    x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps).conjugate() * maps_.constant(inv_sqrt2);
  finishForward(y, time, true);
}

void VCCSENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  Sz6        st0{0, 0, 0, 0, 0, 0}, st1{0, 1, 0, 0, 0, 0}, sz = oshape;
  sz[1] = 1;
  x.device(Threads::GlobalDevice()) += (ConjugateSum(y.slice(st0, sz), maps_.reshape(resMaps).broadcast(brdMaps)) +
                                        ConjugateSum(y.slice(st1, sz), maps_.reshape(resMaps).broadcast(brdMaps).conjugate())) *
                                       maps_.constant(inv_sqrt2);
  finishAdjoint(x, time, true);
}

auto VCCSENSE::nChannels() const -> Index { return oshape[0]; }
auto VCCSENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl::TOps
