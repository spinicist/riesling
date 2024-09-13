#include "top.hpp"

#include "log.hpp"

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

template <typename S, int I, int O> TOp<S, I, O>::~TOp(){};
template <typename S, int I, int O> auto TOp<S, I, O>::rows() const -> Index { return Product(oshape); }
template <typename S, int I, int O> auto TOp<S, I, O>::cols() const -> Index { return Product(ishape); }

template <typename S, int I, int O> void TOp<S, I, O>::forward(typename Base::CMap const &x, typename Base::Map &y) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::adjoint(typename Base::CMap const &y, typename Base::Map &x) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm);
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(typename Base::CMap const &x, typename Base::Map &y) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  iforward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(typename Base::CMap const &y, typename Base::Map &x) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  iadjoint(ym, xm);
}

template <typename S, int I, int O> auto TOp<S, I, O>::forward(InTensor const &x) const -> OutTensor
{
  assert(x.dimensions() == ishape);
  InCMap    xm(x.data(), ishape);
  OutTensor y(oshape);
  OutMap    ym(y.data(), oshape);
  Log::Debug("TOp", "{} forward allocated {}", this->name, ym.dimensions());
  forward(xm, ym);
  return y;
}

template <typename S, int I, int O> auto TOp<S, I, O>::adjoint(OutTensor const &y) const -> InTensor
{
  assert(y.dimensions() == oshape);
  OutCMap  ym(y.data(), oshape);
  InTensor x(ishape);
  InMap    xm(x.data(), ishape);
  Log::Debug("TOp", "{} adjoint allocated {}", this->name, xm.dimensions());
  adjoint(ym, xm);
  return x;
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(InCMap const &, OutMap &) const
{
  throw Log::Failure("TOp", "In place {} not implemented", this->name);
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(OutCMap const &, InMap &) const
{
  throw Log::Failure("TOp", "In place {} not implemented", this->name);
}

template <typename S, int I, int O>
auto TOp<S, I, O>::startForward(InCMap const &x, OutMap const &y, bool const ip) const -> Time
{
  if (x.dimensions() != ishape) { throw Log::Failure("TOp", "{} forward x dims: {} expected: {}", this->name, x.dimensions(), ishape); }
  if (y.dimensions() != oshape) { throw Log::Failure("TOp", "{} forward y dims: {} expected: {}", this->name, y.dimensions(), oshape); }
  Log::Debug("TOp", "{} {}forward {}->{} |x| {}", this->name, (ip ? "IP " : ""), this->ishape, this->oshape, Norm(x));
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishForward(OutMap const &y, Time const start, bool const ip) const
{
  Log::Debug("TOp", "{} {}forward finished in {} |y| {}.", this->name, (ip ? "IP " : ""), Log::ToNow(start), Norm(y));
}

template <typename S, int I, int O>
auto TOp<S, I, O>::startAdjoint(OutCMap const &y, InMap const &x, bool const ip) const -> Time
{
  if (y.dimensions() != oshape) { throw Log::Failure("TOp", "{} adjoint y dims: {} expected: {}", this->name, y.dimensions(), oshape); }
  if (x.dimensions() != ishape) { throw Log::Failure("TOp", "{} adjoint x dims: {} expected: {}", this->name, x.dimensions(), ishape); }
  Log::Debug("TOp", "{} {}adjoint {}->{} |y| {}", this->name, (ip ? "IP " : ""), this->oshape, this->ishape, Norm(y));
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishAdjoint(InMap const &x, Time const start, bool const ip) const
{
  Log::Debug("TOp", "{} {}adjoint finished in {} |x| {}", this->name, (ip ? "IP " : ""), Log::ToNow(start), Norm(x));
}

// Yeah, this was likely a mistake
template struct TOp<Cx, 1, 1>;
template struct TOp<Cx, 2, 2>;
template struct TOp<Cx, 3, 3>;
template struct TOp<Cx, 4, 3>;
template struct TOp<Cx, 4, 4>;
template struct TOp<Cx, 4, 5>;
template struct TOp<Cx, 4, 6>;
template struct TOp<Cx, 5, 3>;
template struct TOp<Cx, 5, 4>;
template struct TOp<Cx, 5, 5>;
template struct TOp<Cx, 5, 6>;
template struct TOp<Cx, 5, 7>;
template struct TOp<Cx, 6, 3>;
template struct TOp<Cx, 6, 4>;
template struct TOp<Cx, 6, 5>;
template struct TOp<Cx, 6, 6>;
template struct TOp<Cx, 6, 7>;
template struct TOp<Cx, 7, 4>;

template struct TOp<float, 1, 1>;
template struct TOp<float, 2, 2>;
template struct TOp<float, 3, 3>;
template struct TOp<float, 4, 3>;
template struct TOp<float, 4, 4>;
template struct TOp<float, 5, 3>;
template struct TOp<float, 5, 5>;
template struct TOp<float, 5, 7>;

template <typename S, int R>
Identity<S, R>::Identity(Sz<R> dims)
  : Parent("Identity", dims, dims)
{
}

template <typename S, int R> void Identity<S, R>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = Parent::startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x;
  Parent::finishAdjoint(y, time, false);
}

template <typename S, int R> void Identity<S, R>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = Parent::startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y;
  Parent::finishAdjoint(x, time, false);
}

template <typename S, int R> void Identity<S, R>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = Parent::startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x;
  Parent::finishAdjoint(y, time, true);
}

template <typename S, int R> void Identity<S, R>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = Parent::startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y;
  Parent::finishAdjoint(x, time, true);
}

template struct Identity<Cx, 4>;
template struct Identity<Cx, 5>;

} // namespace rl::TOps
