#include "sense.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND> SENSE<ND>::SENSE(CxN<ND + 2> const &maps, Index const nB)
  : Parent("SENSEOp", AddBack(FirstN<ND>(maps.dimensions()), nB), AddBack(FirstN<ND>(maps.dimensions()), maps.dimension(ND), nB))
  , maps_{maps}
{
  for (int ii = 0; ii < ND; ii++) {
    resX.set(ii, maps_.dimension(ii));
  }
  resX.set(ND + 1, nB);
  brdX.set(ND, maps_.dimension(ND));

  if (maps_.dimension(ND + 1) == 1) {
    brdMaps.set(ND + 1, nB);
  } else if (maps_.dimension(0) == nB) {
    brdMaps.set(ND + 1, 1);
  } else {
    throw Log::Failure("SENSE", "Maps had basis size {} expected {}", maps_.dimension(ND + 1), nB);
  }
}

template <int ND> void SENSE<ND>::forward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  this->finishForward(y, time, false);
}

template <int ND> void SENSE<ND>::iforward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  this->finishForward(y, time, true);
}

template <int ND> void SENSE<ND>::adjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  this->finishAdjoint(x, time, false);
}

template <int ND> void SENSE<ND>::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  this->finishAdjoint(x, time, true);
}

template <int ND> auto SENSE<ND>::nChannels() const -> Index { return oshape[ND]; }
template <int ND> auto SENSE<ND>::mapDimensions() const -> Sz<ND> { return FirstN<ND>(ishape); }

template struct SENSE<2>;
template struct SENSE<3>;
} // namespace rl::TOps
