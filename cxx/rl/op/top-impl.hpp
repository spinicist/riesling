#pragma once

#include "top.hpp"

#include "../log/log.hpp"

namespace rl::TOps {

template <int I, int O> TOp<I, O>::TOp(std::string const &n)
  : Ops::Op{n}
{
  Log::Debug("TOp", "{} created.", this->name);
}

template <int I, int O> TOp<I, O>::TOp(std::string const &n, InDims const xd, OutDims const yd)
  : Ops::Op{n}
  , ishape{xd}
  , oshape{yd}
{
  Log::Debug("TOp", "{} created. Input dims {} Output dims {}", this->name, ishape, oshape);
}

template <int I, int O> auto TOp<I, O>::rows() const -> Index { return Product(oshape); }
template <int I, int O> auto TOp<I, O>::cols() const -> Index { return Product(ishape); }

template <int I, int O> void TOp<I, O>::forward(typename Base::CMap x, typename Base::Map y, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym, s);
}

template <int I, int O> void TOp<I, O>::adjoint(typename Base::CMap y, typename Base::Map x, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm, s);
}

template <int I, int O> void TOp<I, O>::inverse(typename Base::CMap y, typename Base::Map x, float const s, float const b) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  inverse(ym, xm, s, b);
}

template <int I, int O> void TOp<I, O>::iforward(typename Base::CMap x, typename Base::Map y, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  iforward(xm, ym, s);
}

template <int I, int O> void TOp<I, O>::iadjoint(typename Base::CMap y, typename Base::Map x, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  iadjoint(ym, xm, s);
}

template <int I, int O> auto TOp<I, O>::forward(InTensor const &x, float const s) const -> OutTensor
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  InCMap    xm(x.data(), ishape);
  OutTensor y(oshape);
  OutMap    ym(y.data(), oshape);
  Log::Debug("TOp", "{} forward allocated {}", this->name, ym.dimensions());
  forward(xm, ym, s);
  return y;
}

template <int I, int O> auto TOp<I, O>::adjoint(OutTensor const &y, float const s) const -> InTensor
{
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  OutCMap  ym(y.data(), oshape);
  InTensor x(ishape);
  InMap    xm(x.data(), ishape);
  Log::Debug("TOp", "{} adjoint allocated {}", this->name, xm.dimensions());
  adjoint(ym, xm, s);
  return x;
}

template <int I, int O> void TOp<I, O>::forward(InTensor const &x, OutTensor &y, float const s) const
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym, s);
}

template <int I, int O> void TOp<I, O>::adjoint(OutTensor const &y, InTensor &x, float const s) const
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm, s);
}

template <int I, int O> void TOp<I, O>::inverse(OutCMap y, InMap x, float, float) const
{
  throw Log::Failure(this->name, "Inverse not implemented");
}

template <int I, int O> void TOp<I, O>::iforward(InCMap, OutMap, float) const
{
  throw Log::Failure(this->name, "In place not implemented");
}

template <int I, int O> void TOp<I, O>::iadjoint(OutCMap, InMap, float) const
{
  throw Log::Failure(this->name, "In place not implemented");
}

template <int I, int O> auto TOp<I, O>::startForward(InCMap x, OutMap y, bool const ip) const -> Time
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "Forward x dims: {} expected: {}", x.dimensions(), ishape); }
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "Forward y dims: {} expected: {}", y.dimensions(), oshape); }
  if (Log::IsHigh()) {
    if (ip) {
      Log::Debug(this->name, "IP Forward {}->{} |x| {} |y| {}", ishape, oshape, Norm<true>(x), Norm<true>(y));
    } else {
      Log::Debug(this->name, "Forward {}->{} |x| {}", ishape, oshape, Norm<true>(x));
    }
  } else {
    Log::Debug(this->name, "{}Forward {}->{}", (ip ? "IP " : ""), ishape, oshape);
  }
  return Log::Now();
}

template <int I, int O> void TOp<I, O>::finishForward(OutMap y, Time const start, bool const ip) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Forward finished in {} |y| {}", (ip ? "IP " : ""), Log::ToNow(start), Norm<true>(y));
  } else {
    Log::Debug(this->name, "{}Forward finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <int I, int O> auto TOp<I, O>::startAdjoint(OutCMap y, InMap x, bool const ip) const -> Time
{
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "Adjoint y dims: {} expected: {}", y.dimensions(), oshape); }
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "Adjoint x dims: {} expected: {}", x.dimensions(), ishape); }
  if (Log::IsHigh()) {
    if (ip) {
      Log::Debug(this->name, "IP Adjoint {}->{} |y| {} |x| {}", oshape, ishape, Norm<true>(y), Norm<true>(x));
    } else {
      Log::Debug(this->name, "Adjoint {}->{} |y| {}", oshape, ishape, Norm<true>(y));
    }
  } else {
    Log::Debug(this->name, "{}Adjoint {}->{}", (ip ? "IP " : ""), oshape, ishape);
  }
  return Log::Now();
}

template <int I, int O> void TOp<I, O>::finishAdjoint(InMap x, Time const start, bool const ip) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Adjoint finished in {} |x| {}", (ip ? "IP " : ""), Log::ToNow(start), Norm<true>(x));
  } else {
    Log::Debug(this->name, "{}Adjoint finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <int I, int O> auto TOp<I, O>::startInverse(OutCMap y, InMap x) const -> Time
{
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "Inverse y dims: {} expected: {}", y.dimensions(), oshape); }
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "Inverse x dims: {} expected: {}", x.dimensions(), ishape); }
  if (Log::IsHigh()) {
    Log::Debug(this->name, "Inverse {}->{} |y| {}", oshape, ishape, Norm<true>(y));
  } else {
    Log::Debug(this->name, "Inverse {}->{}", oshape, ishape);
  }
  return Log::Now();
}

template <int I, int O> void TOp<I, O>::finishInverse(InMap x, Time const start) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "Inverse finished in {} |x| {}", Log::ToNow(start), Norm<true>(x));
  } else {
    Log::Debug(this->name, "Inverse finished in {}", Log::ToNow(start));
  }
}

} // namespace rl::TOps
