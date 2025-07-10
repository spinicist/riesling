#include "mask.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Ops {

Mask::Mask(MaskVector const &m, Index const r)
  : Op("Mask")
  , mask{m}
  , repeats{r}
  , isz{mask.rows()}
  , osz{0}
{
  for (Index ii = 0; ii < mask.rows(); ii++) {
    if (mask[ii] > 0.f) { osz++; }
  }
}

auto Mask::Make(MaskVector const &m, Index const r) -> Ptr { return std::make_shared<Mask>(m, r); }
auto Mask::rows() const -> Index { return osz * repeats; }
auto Mask::cols() const -> Index { return isz * repeats; }
void Mask::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { y[iy++] = x[ix] * s; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishForward(y, time, false);
}

void Mask::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) {
        x[ix] = y[iy++] * s;
      } else {
        x[ix] = 0.f;
      }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishAdjoint(x, time, false);
}

void Mask::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { y[iy++] += x[ix] * s; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishForward(y, time, true);
}

void Mask::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ix = 0, iy = 0;
  for (Index ir = 0; ir < repeats; ir++) {
    for (Index im = 0; im < isz; im++, ix++) {
      if (mask[im]) { x[ix] += y[iy++] * s; }
    }
  }
  if (ix != cols()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  if (iy != rows()) { throw Log::Failure(this->name, "Mask logic incorrect"); }
  this->finishAdjoint(x, time, true);
}

} // namespace rl::Ops
