#include "top.hpp"

namespace rl::TOps {

template <typename S, int I, int O>
TOp<S, I, O>::TOp(std::string const &n)
  : Ops::Op<Scalar>{n}
{
  Log::Debug("{} created.", this->name);
}

template <typename S, int I, int O>
TOp<S, I, O>::TOp(std::string const &n, InDims const xd, OutDims const yd)
  : Ops::Op<Scalar>{n}
  , ishape{xd}
  , oshape{yd}
{
  Log::Debug("{} created. Input dims {} Output dims {}", this->name, ishape, oshape);
}

template <typename S, int I, int O> TOp<S, I, O>::~TOp(){};
template <typename S, int I, int O> auto TOp<S, I, O>::rows() const -> Index { return Product(oshape); }
template <typename S, int I, int O> auto TOp<S, I, O>::cols() const -> Index { return Product(ishape); }

template <typename S, int I, int O> void TOp<S, I, O>::forward(typename Base::CMap const &x, typename Base::Map &y) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  Log::Debug("TOp {} forward {}->{}", this->name, this->ishape, this->oshape);
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  forward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::adjoint(typename Base::CMap const &y, typename Base::Map &x) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  Log::Debug("TOp {} adjoint {}->{}", this->name, this->oshape, this->ishape);
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  adjoint(ym, xm);
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(typename Base::CMap const &x, typename Base::Map &y) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  Log::Debug("TOp {} forward {}->{}", this->name, this->ishape, this->oshape);
  InCMap xm(x.data(), ishape);
  OutMap ym(y.data(), oshape);
  iforward(xm, ym);
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(typename Base::CMap const &y, typename Base::Map &x) const
{
  assert(x.rows() == this->cols());
  assert(y.rows() == this->rows());
  Log::Debug("TOp {} adjoint {}->{}", this->name, this->oshape, this->ishape);
  OutCMap ym(y.data(), oshape);
  InMap   xm(x.data(), ishape);
  iadjoint(ym, xm);
}

template <typename S, int I, int O> auto TOp<S, I, O>::forward(InTensor const &x) const -> OutTensor
{
  InCMap    xm(x.data(), ishape);
  OutTensor y(oshape);
  OutMap    ym(y.data(), oshape);
  Log::Debug("TOp {} forward {}->{} Allocated {}", this->name, this->ishape, this->oshape, ym.dimensions());
  forward(xm, ym);
  return y;
}

template <typename S, int I, int O> auto TOp<S, I, O>::adjoint(OutTensor const &y) const -> InTensor
{
  OutCMap  ym(y.data(), oshape);
  InTensor x(ishape);
  InMap    xm(x.data(), ishape);
  Log::Debug("TOp {} adjoint {}->{} Allocated {}", this->name, this->oshape, this->ishape, xm.dimensions());
  adjoint(ym, xm);
  return x;
}

template <typename S, int I, int O> void TOp<S, I, O>::iforward(InCMap const &x, OutMap &y) const
{
  Log::Fail("In place {} not implemented", this->name);
}

template <typename S, int I, int O> void TOp<S, I, O>::iadjoint(OutCMap const &y, InMap &x) const
{
  Log::Fail("In place {} not implemented", this->name);
}

template <typename S, int I, int O> auto TOp<S, I, O>::startForward(InCMap const &x, OutMap const &y) const -> Log::Time
{
  if (x.dimensions() != ishape) { Log::Fail("{} forward x dims were: {} expected: {}", this->name, x.dimensions(), ishape); }
  if (y.dimensions() != oshape) { Log::Fail("{} forward y dims were: {} expected: {}", this->name, y.dimensions(), oshape); }
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("{} forward started. Dimensions {}->{}. Norm {}", this->name, this->ishape, this->oshape, Norm(x));
  }
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishForward(OutMap const &y, Log::Time const start) const
{
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("{} forward finished. Took {}. Norm {}.", this->name, Log::ToNow(start), Norm(y));
  }
}

template <typename S, int I, int O> auto TOp<S, I, O>::startAdjoint(OutCMap const &y, InMap const &x) const -> Log::Time
{
  if (y.dimensions() != oshape) { Log::Fail("{} adjoint y dims were: {} expected: {}", this->name, y.dimensions(), oshape); }
  if (x.dimensions() != ishape) { Log::Fail("{} adjoint x dims were: {} expected: {}", this->name, x.dimensions(), ishape); }
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("{} adjoint started. Dimensions {}->{}. Norm {}", this->name, this->oshape, this->ishape, Norm(y));
  }
  return Log::Now();
}

template <typename S, int I, int O> void TOp<S, I, O>::finishAdjoint(InMap const &x, Log::Time const start) const
{
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("{} adjoint finished. Took {}. Norm {}", this->name, Log::ToNow(start), Norm(x));
  }
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
template struct TOp<Cx, 6, 6>;
template struct TOp<Cx, 6, 7>;
template struct TOp<Cx, 7, 4>;

template struct TOp<float, 1, 1>;
template struct TOp<float, 2, 2>;
template struct TOp<float, 3, 3>;
template struct TOp<float, 4, 3>;
template struct TOp<float, 5, 3>;
template struct TOp<float, 5, 7>;

template <typename S, int R>
Identity<S, R>::Identity(Sz<R> dims)
  : Parent("Identity", dims, dims)
{
}

template <typename S, int R> void Identity<S, R>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = Parent::startForward(x, y);
  y.device(Threads::GlobalDevice()) = x;
  Parent::finishAdjoint(y, time);
}

template <typename S, int R> void Identity<S, R>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = Parent::startAdjoint(y, x);
  x.device(Threads::GlobalDevice()) = y;
  Parent::finishAdjoint(x, time);
}

template <typename S, int R> void Identity<S, R>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = Parent::startForward(x, y);
  y.device(Threads::GlobalDevice()) += x;
  Parent::finishAdjoint(y, time);
}

template <typename S, int R> void Identity<S, R>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = Parent::startAdjoint(y, x);
  x.device(Threads::GlobalDevice()) += y;
  Parent::finishAdjoint(x, time);
}

template struct Identity<Cx, 4>;
template struct Identity<Cx, 5>;

} // namespace rl::TOps
