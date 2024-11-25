#include "sense.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

#include <numbers>
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;

namespace rl::TOps {

SENSE::SENSE(Cx5 const &maps, bool const v, Index const nB)
  : Parent("SENSEOp",
           AddFront(LastN<3>(maps.dimensions()), nB),
           AddFront(LastN<3>(maps.dimensions()), nB, maps.dimension(1) * (v ? 2 : 1)))
  , maps_{std::move(maps)}
  , vcc{v}
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
    throw Log::Failure("SENSE", "Maps had basis size {} expected {}", maps_.dimension(0), nB);
  }
}

void SENSE::forward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, false);
  if (vcc) {
    Index const nC = y.dimension(1) / 2;
    Sz5         sz = y.dimensions();
    sz[1] = nC;
    y.slice(Sz5{0, 0, 0, 0, 0}, sz).device(Threads::TensorDevice()) =
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
    y.slice(Sz5{0, nC, 0, 0, 0}, sz).device(Threads::TensorDevice()) =
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  } else {
    y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  }
  finishForward(y, time, false);
}

void SENSE::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = startForward(x, y, true);
  if (vcc) {
    Index const nC = y.dimension(0) / 2;
    Sz5         sz = y.dimensions();
    sz[1] = nC;
    y.slice(Sz5{0, 0, 0, 0, 0}, sz).device(Threads::TensorDevice()) +=
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * Cx(inv_sqrt2);
    y.slice(Sz5{0, nC, 0, 0, 0}, sz).device(Threads::TensorDevice()) +=
      x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps).conjugate() * Cx(inv_sqrt2);
  } else {
    y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  }
  finishForward(y, time, true);
}

void SENSE::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, false);
  if (vcc) {
    Index const nC = y.dimension(1) / 2;
    Sz5         sz = y.dimensions();
    sz[1] = nC;
    auto const real = (y.slice(Sz5{0, 0, 0, 0, 0}, sz) * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{1});
    auto const virt = (y.slice(Sz5{0, nC, 0, 0, 0}, sz) * maps_.broadcast(brdMaps)).sum(Sz1{1});
    x.device(Threads::TensorDevice()) = (real + virt) * Cx(inv_sqrt2);
  } else {
    x.device(Threads::TensorDevice()) = (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{1});
  }
  finishAdjoint(x, time, false);
}

void SENSE::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = startAdjoint(y, x, true);
  if (vcc) {
    Index const nC = y.dimension(1) / 2;
    Sz5         sz = y.dimensions();
    sz[1] = nC;
    auto const real = (y.slice(Sz5{0, 0, 0, 0, 0}, sz) * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{1});
    auto const virt = (y.slice(Sz5{0, nC, 0, 0, 0}, sz) * maps_.broadcast(brdMaps)).sum(Sz1{1});
    x.device(Threads::TensorDevice()) += (real + virt) * Cx(inv_sqrt2);
  } else {
    x.device(Threads::TensorDevice()) += (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{1});
  }
  finishAdjoint(x, time, true);
}

auto SENSE::nChannels() const -> Index { return oshape[1]; }
auto SENSE::mapDimensions() const -> Sz3 { return LastN<3>(ishape); }

} // namespace rl::TOps
