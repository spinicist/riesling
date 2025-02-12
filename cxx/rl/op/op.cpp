#include "op.hpp"

#include "../algo/common.hpp"
#include "../log.hpp"
#include "../tensors.hpp"
#include "ops.hpp"

namespace rl::Ops {

template <typename S>
Op<S>::Op(std::string const &n)
  : name{n}
{
}

template <typename S> void Op<S>::inverse(CMap const , Map ) const
{
  throw Log::Failure("Op", "{} does not have an inverse defined", name);
}

template <typename S> void Op<S>::forward(Vector const &x, Vector &y) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->forward(xm, ym);
}

template <typename S> void Op<S>::adjoint(Vector const &y, Vector &x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->adjoint(ym, xm);
}

template <typename S> void Op<S>::inverse(Vector const &y, Vector &x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->inverse(ym, xm);
}

template <typename S> auto Op<S>::forward(Vector const &x) const -> Vector
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  Vector y(this->rows());
  Map    ym(y.data(), y.size());
  y.setZero();
  Log::Debug("Op", "{} forward allocated [{}]", name, y.size());
  this->forward(CMap(x.data(), x.size()), ym);
  return y;
}

template <typename S> auto Op<S>::adjoint(Vector const &y) const -> Vector
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  Vector x(this->cols());
  Map    xm(x.data(), x.size());
  x.setZero();
  Log::Debug("Op", "{} adjoint allocated [{}]", name, x.size());
  this->adjoint(CMap(y.data(), y.size()), xm);
  return x;
}

template <typename S> void Op<S>::iforward(Vector const &x, Vector &y) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->iforward(xm, ym);
}

template <typename S> void Op<S>::iadjoint(Vector const &y, Vector &x) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->iadjoint(ym, xm);
}

template <typename S> auto Op<S>::startForward(CMap const &x, Map const &y, bool const ip) const -> Log::Time
{
  if (x.rows() != cols()) { throw Log::Failure("Op", "{} forward x [{}] expected [{}]", this->name, x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure("Op", "{} forward y [{}] expected [{}]", this->name, y.rows(), rows()); }
  if (Log::IsDebugging()) {
    Log::Debug("Op", "{} {}forward [{}, {}] |x| {}", this->name, (ip ? "IP " : ""), rows(), cols(), ParallelNorm(x));
  } else {
    Log::Debug("Op", "{} {}forward [{}, {}]", this->name, (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishForward(Map const &y, Log::Time const start, bool const ip) const
{
  Log::Debug("Op", "{} {}forward finished in {} |y| {}.", this->name, (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(y));
}

template <typename S> auto Op<S>::startAdjoint(CMap const &y, Map const &x, bool const ip) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure("Op", "{} adjoint y [{}] expected [{}]", this->name, y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure("Op", "{} adjoint x [{}] expected [{}]", this->name, x.rows(), cols()); }
  if (Log::IsDebugging()) {
    Log::Debug("Op", "{} {}adjoint [{},{}] |y| {}", this->name, (ip ? "IP " : ""), rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug("Op", "{} {}adjoint [{},{}]", this->name, (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishAdjoint(Map const &x, Log::Time const start, bool const ip) const
{
  if (Log::IsDebugging()) {
    Log::Debug("Op", "{} {}adjoint finished in {} |x| {}", this->name, (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug("Op", "{} {}adjoint finished in {}", this->name, (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <typename S> auto Op<S>::startInverse(CMap const &y, Map const &x, bool const ip) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure("Op", "{} inverse y [{}] expected [{}]", this->name, y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure("Op", "{} inverse x [{}] expected [{}]", this->name, x.rows(), cols()); }
  if (Log::IsDebugging()) {
    Log::Debug("Op", "{} {}inverse [{},{}] |y| {}", this->name, (ip ? "IP " : ""), rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug("Op", "{} {}inverse [{},{}]", this->name, (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

template <typename S> void Op<S>::finishInverse(Map const &x, Log::Time const start, bool const ip) const
{
  if (Log::IsDebugging()) {
    Log::Debug("Op", "{} {}inverse finished in {} |x| {}", this->name, (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug("Op", "{} {}inverse finished in {}", this->name, (ip ? "IP " : ""), Log::ToNow(start));
  }
}

template <typename S> auto Op<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  throw Log::Failure("Op", "{} does not have an inverse", name);
}

template <typename S> auto Op<S>::inverse(float const, float const) const -> std::shared_ptr<Op<S>>
{
  throw Log::Failure("Op", "{} does not have an inverse", name);
}

template <typename S> auto Op<S>::operator+(S const) const -> std::shared_ptr<Op<S>>
{
  throw Log::Failure("Op", "{} does not have operator+", name);
}

template struct Op<float>;
template struct Op<Cx>;

} // namespace rl::Ops
