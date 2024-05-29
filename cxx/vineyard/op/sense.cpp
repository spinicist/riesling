#include "sense.hpp"

#include "tensors.hpp"
#include "threads.hpp"

#include <numbers>

constexpr float inv_sqrt2 = 1.f / std::numbers::sqrt2;

namespace rl::TOps {

SENSE::SENSE(Cx5 const &maps, Index const frames, bool const vcc)
  : Parent("SENSEOp",
           AddFront(LastN<3>(maps.dimensions()), frames),
           AddFront(LastN<3>(maps.dimensions()), maps.dimension(0) * (vcc ? 2 : 1), frames))
  , maps_{std::move(maps)}
  , vcc_{vcc}
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

void SENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y);
  if (vcc_) {
    Sz5 st{0, 0, 0, 0, 0}, sz = oshape;
    sz[0] /= 2;
    y.slice(st, sz).device(Threads::GlobalDevice()) =
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * maps_.constant(inv_sqrt2);
    st[0] += sz[0];
    y.slice(st, sz).device(Threads::GlobalDevice()) =
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * maps_.constant(inv_sqrt2);
  } else {
    y.device(Threads::GlobalDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  }
  finishForward(y, time);
}

void SENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x);
  if (vcc_) {
    Sz5 st{0, 0, 0, 0, 0}, sz = oshape;
    sz[0] /= 2;
    x.device(Threads::GlobalDevice()) = ConjugateSum(y.slice(st, sz), maps_.broadcast(brdMaps)) * maps_.constant(inv_sqrt2);
    st[0] += sz[0];
    x.device(Threads::GlobalDevice()) =
      x + ConjugateSum(y.slice(st, sz), maps_.broadcast(brdMaps).conjugate()) * maps_.constant(inv_sqrt2);
  } else {
    x.device(Threads::GlobalDevice()) = ConjugateSum(y, maps_.broadcast(brdMaps));
  }
  finishAdjoint(x, time);
}

auto SENSE::nChannels() const -> Index { return oshape[0]; }
auto SENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl::TOps
