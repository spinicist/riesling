#include "mask.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Ops {

template <typename S>
Mask<S>::Mask(MaskVector const &m, Index const r)
  : Op<S>("Mask")
  , mask{m}
  , repeats{r}
  , isz{mask.rows()}
  , osz{0}
{
  for (Index ii = 0; ii < mask.rows(); ii++) {
    if (mask[ii] > 0.f) { osz++; }
  }
}

template <typename S> auto Mask<S>::rows() const -> Index { return osz * repeats; }
template <typename S> auto Mask<S>::cols() const -> Index { return isz * repeats; }

template <typename S> void Mask<S>::forward(CMap x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { y[iy++] = x[ix]; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishForward(y, time, false);
}

template <typename S> void Mask<S>::adjoint(CMap y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) {
        x[ix] = y[iy++];
      } else {
        x[ix] = 0.f;
      }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishAdjoint(x, time, false);
}

template <typename S> void Mask<S>::iforward(CMap x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { y[iy++] += x[ix]; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishForward(y, time, true);
}

template <typename S> void Mask<S>::iadjoint(CMap y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { x[ix] += y[iy++]; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishAdjoint(x, time, true);
}

template struct Mask<float>;
template struct Mask<Cx>;

} // namespace rl::Ops
