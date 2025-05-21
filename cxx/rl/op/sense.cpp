#include "sense.hpp"

#include "../log.hpp"
#include "../sense/sense.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND> SENSEOp<ND>::SENSEOp(CxN<ND + 2> const &maps, Index const nB)
  : Parent(
      "SENSEOp", AddBack(FirstN<ND>(maps.dimensions()), nB), AddBack(FirstN<ND>(maps.dimensions()), maps.dimension(ND), nB))
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
    throw Log::Failure("SENSEOp", "Maps had basis size {} expected {}", maps_.dimension(ND + 1), nB);
  }
}

template <int ND> SENSEOp<ND>::SENSEOp(CxN<ND + 2> const &kern, Sz<ND> const mat, float const os, Index const nB)
  : Parent("SENSEOp", AddBack(mat, nB), AddBack(mat, kern.dimension(ND), nB))
  , maps_{SENSE::KernelsToMaps(kern, mat, os)}
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
    throw Log::Failure("SENSEOp", "Kernels had basis size {} expected {}", maps_.dimension(ND + 1), nB);
  }
}

template <int ND> void SENSEOp<ND>::forward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  this->finishForward(y, time, false);
}

template <int ND> void SENSEOp<ND>::iforward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps);
  this->finishForward(y, time, true);
}

template <int ND> void SENSEOp<ND>::adjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  this->finishAdjoint(x, time, false);
}

template <int ND> void SENSEOp<ND>::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += (y * maps_.broadcast(brdMaps).conjugate()).sum(Sz1{3});
  this->finishAdjoint(x, time, true);
}

template <int ND> auto SENSEOp<ND>::nChannels() const -> Index { return oshape[ND]; }
template <int ND> auto SENSEOp<ND>::mapDimensions() const -> Sz<ND> { return FirstN<ND>(ishape); }
template <int ND> auto SENSEOp<ND>::maps() const -> CxN<ND + 2> { return maps_; }

template struct SENSEOp<2>;
template struct SENSEOp<3>;

template <int ND> auto MakeSENSE(CxN<ND + 2> const &maps, Index const nB) -> SENSEOp<ND>::Ptr
{
  return std::make_shared<SENSEOp<ND>>(maps, nB);
}
template <int ND> auto MakeSENSE(CxN<ND + 2> const &kern, Sz<ND> const mat, float const os, Index const nB) -> SENSEOp<ND>::Ptr
{
  return std::make_shared<SENSEOp<ND>>(kern, mat, os, nB);
}

template auto MakeSENSE<3>(Cx5 const &, Index const) -> SENSEOp<3>::Ptr;
template auto MakeSENSE<3>(Cx5 const &, Sz3 const, float const, Index const) -> SENSEOp<3>::Ptr;

} // namespace rl::TOps
