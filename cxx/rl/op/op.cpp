#include "op.hpp"

#include "../algo/common.hpp"
#include "../log/debug.hpp"
#include "../tensors.hpp"
#include "ops.hpp"

namespace rl::Ops {

template <typename S> Op<S>::Op(std::string const &n)
  : name{n}
{
}

template <typename S> void Op<S>::inverse(CMap, Map, float const, float const) const
{
  throw Log::Failure(this->name, "Does not have an inverse defined", name);
}

template <typename S> void Op<S>::forward(Vector const &x, Vector &y) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->forward(xm, ym);
}

template <typename S> void Op<S>::adjoint(Vector const &y, Vector &x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->adjoint(ym, xm);
}

template <typename S> void Op<S>::inverse(Vector const &y, Vector &x, float const s, float const b) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Inverse x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Inverse y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->inverse(ym, xm, s, b);
}

template <typename S> auto Op<S>::forward(Vector const &x) const -> Vector
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x {} != cols {}", x.rows(), cols()); }
  Vector y(this->rows());
  Map    ym(y.data(), y.size());
  y.setZero();
  Log::Debug(this->name, "Forward allocated [{}]", y.size());
  this->forward(CMap(x.data(), x.size()), ym);
  return y;
}

template <typename S> auto Op<S>::adjoint(Vector const &y) const -> Vector
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y {} != rows {}", y.rows(), rows()); }
  Vector x(this->cols());
  Map    xm(x.data(), x.size());
  x.setZero();
  Log::Debug(this->name, "Adjoint allocated [{}]", x.size());
  this->adjoint(CMap(y.data(), y.size()), xm);
  return x;
}

template <typename S> void Op<S>::iforward(Vector const &x, Vector &y, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward+ x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward+ y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->iforward(xm, ym, s);
}

template <typename S> void Op<S>::iadjoint(Vector const &y, Vector &x, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint+ x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint+ y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->iadjoint(ym, xm, s);
}

template <typename S> auto Op<S>::startForward(CMap x, Map const &y, bool const ip) const -> Log::Time
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x [{}] expected [{}]", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward y [{}] expected [{}]", y.rows(), rows()); }
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}forward [{}, {}] |x| {}", (ip ? "IP " : ""), rows(), cols(), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "{}forward [{}, {}]", (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishForward(Map const &y, Log::Time const start, bool const ip) const
{
  Log::Debug(this->name, "{}forward finished in {} |y| {}.", (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(y));
}

template <typename S> auto Op<S>::startAdjoint(CMap y, Map const &x, bool const ip) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y [{}] expected [{}]", y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint x [{}] expected [{}]", x.rows(), cols()); }
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}adjoint [{},{}] |y| {}", (ip ? "IP " : ""), rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug(this->name, "{}adjoint [{},{}]", (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishAdjoint(Map const &x, Log::Time const start, bool const ip) const
{
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "{}adjoint finished in {} |x| {}", (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "{}adjoint finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <typename S> auto Op<S>::startInverse(CMap y, Map const &x) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Inverse y [{}] expected [{}]", y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Inverse x [{}] expected [{}]", x.rows(), cols()); }
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "Inverse [{},{}] |y| {}", rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug(this->name, "Inverse [{},{}]", rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishInverse(Map const &x, Log::Time const start) const
{
  if (Log::IsDebugging()) {
    Log::Debug(this->name, "Inverse finished in {} |x| {}", Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "Inverse finished in {}", Log::ToNow(start));
  }
}

template struct Op<float>;
template struct Op<Cx>;

} // namespace rl::Ops
