#pragma once

#include "top.hpp"

#include "../log/debug.hpp"

namespace rl::TOps {

template <typename S, int I, int O>
TOp<S, I, O>::TOp(std::string const &n)
  : Ops::Op<Scalar>{n}
{
  Log::Debug("TOp", "{} created.", this->name);
}

template <typename S, int I, int O>
TOp<S, I, O>::TOp(std::string const &n, InDims const xd, OutDims const yd)
  : Ops::Op<Scalar>{n}
  , ishape{xd}
  , oshape{yd}
{
  Log::Debug("TOp", "{} created. Input dims {} Output dims {}", this->name, ishape, oshape);
}

template <typename S, int I, int O> auto TOp<S, I, O>::rows() const -> Index { return Product(oshape); }
template <typename S, int I, int O> auto TOp<S, I, O>::cols() const -> Index { return Product(ishape); }

template <typename S, int I, int O> void TOp<S, I, O>::forward(typename Base::CMap const x, typename Base::Map y) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::adjoint(typename Base::CMap const y, typename Base::Map x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm);
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(typename Base::CMap const x, typename Base::Map y) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  iforward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(typename Base::CMap const y, typename Base::Map x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  iadjoint(ym, xm);
}

template <typename S, int I, int O> auto TOp<S, I, O>::forward(InTensor const &x) const -> OutTensor
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  InCMap    xm(x.data(), ishape);
  OutTensor y(oshape);
  OutMap    ym(y.data(), oshape);
  Log::Debug("TOp", "{} forward allocated {}", this->name, ym.dimensions());
  forward(xm, ym);
  return y;
}

template <typename S, int I, int O> auto TOp<S, I, O>::adjoint(OutTensor const &y) const -> InTensor
{
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  OutCMap  ym(y.data(), oshape);
  InTensor x(ishape);
  InMap    xm(x.data(), ishape);
  Log::Debug("TOp", "{} adjoint allocated {}", this->name, xm.dimensions());
  adjoint(ym, xm);
  return x;
}

template <typename S, int I, int O> void TOp<S, I, O>::forward(InTensor const &x, OutTensor &y) const
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::adjoint(OutTensor const &y, InTensor &x) const
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "xshape {} != ishape {}", x.dimensions(), ishape); };
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "yshape {} != oshape {}", y.dimensions(), oshape); };
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm);
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(InCMap const, OutMap) const
{
  throw Log::Failure(this->name, "In place not implemented");
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(OutCMap const, InMap) const
{
  throw Log::Failure(this->name, "In place not implemented");
}

template <typename S, int I, int O>
auto TOp<S, I, O>::startForward(InCMap const &x, OutMap const &y, bool const ip) const -> Time
{
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "Forward x dims: {} expected: {}", x.dimensions(), ishape); }
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "Forward y dims: {} expected: {}", y.dimensions(), oshape); }
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}Forward {}->{} |x| {}", (ip ? "IP " : ""), ishape, oshape, Norm<true>(x));
  } else {
    Log::Debug(this->name, "{}Forward {}->{}", (ip ? "IP " : ""), ishape, oshape);
  }
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishForward(OutMap const &y, Time const start, bool const ip) const
{
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}Forward finished in {} |y| {}", (ip ? "IP " : ""), Log::ToNow(start), Norm<true>(y));
  } else {
    Log::Debug(this->name, "{}Forward finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <typename S, int I, int O>
auto TOp<S, I, O>::startAdjoint(OutCMap const &y, InMap const &x, bool const ip) const -> Time
{
  if (y.dimensions() != oshape) { throw Log::Failure(this->name, "Adjoint y dims: {} expected: {}", y.dimensions(), oshape); }
  if (x.dimensions() != ishape) { throw Log::Failure(this->name, "Adjoint x dims: {} expected: {}", x.dimensions(), ishape); }
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}Adjoint {}->{} |y| {}", (ip ? "IP " : ""), oshape, ishape, Norm<true>(y));
  } else {
    Log::Debug(this->name, "{}Adjoint {}->{}", (ip ? "IP " : ""), oshape, ishape);
  }
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishAdjoint(InMap const &x, Time const start, bool const ip) const
{
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}Adjoint finished in {} |x| {}", (ip ? "IP " : ""), Log::ToNow(start), Norm<true>(x));
  } else {
    Log::Debug(this->name, "{}Adjoint finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

} // namespace rl::TOps
