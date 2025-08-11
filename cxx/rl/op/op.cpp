#include "op.hpp"

#include "../algo/common.hpp"
#include "../log/log.hpp"
#include "../tensors.hpp"
#include "ops.hpp"

namespace rl::Ops {

Op::Op(std::string const &n)
  : name{n}
{
}

void Op::inverse(CMap, Map, float const, float const) const
{
  throw Log::Failure(this->name, "Does not have an inverse defined", name);
}

void Op::forward(Vector const &x, Vector &y, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->forward(xm, ym, s);
}

void Op::adjoint(Vector const &y, Vector &x, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->adjoint(ym, xm, s);
}

void Op::inverse(Vector const &y, Vector &x, float const s, float const b) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Inverse x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Inverse y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->inverse(ym, xm, s, b);
}

auto Op::forward(Vector const &x, float const s) const -> Vector
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x {} != cols {}", x.rows(), cols()); }
  Vector y(this->rows());
  Map    ym(y.data(), y.size());
  y.setZero();
  Log::Debug(this->name, "Forward allocated [{}]", y.size());
  this->forward(CMap(x.data(), x.size()), ym, s);
  return y;
}

auto Op::adjoint(Vector const &y, float const s) const -> Vector
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y {} != rows {}", y.rows(), rows()); }
  Vector x(this->cols());
  Map    xm(x.data(), x.size());
  x.setZero();
  Log::Debug(this->name, "Adjoint allocated [{}]", x.size());
  this->adjoint(CMap(y.data(), y.size()), xm, s);
  return x;
}

void Op::iforward(Vector const &x, Vector &y, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward+ x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward+ y {} != rows {}", y.rows(), rows()); }
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->iforward(xm, ym, s);
}

void Op::iadjoint(Vector const &y, Vector &x, float const s) const
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint+ x {} != cols {}", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint+ y {} != rows {}", y.rows(), rows()); }
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->iadjoint(ym, xm, s);
}

auto Op::startForward(CMap x, Map const &y, bool const ip) const -> Log::Time
{
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Forward x [{}] expected [{}]", x.rows(), cols()); }
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Forward y [{}] expected [{}]", y.rows(), rows()); }
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Forward [{}, {}] |x| {}", (ip ? "IP " : ""), rows(), cols(), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "{}Forward [{}, {}]", (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

void Op::finishForward(Map const &y, Log::Time const start, bool const ip) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Forward finished in {} |y| {}", (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(y));
  } else {
    Log::Debug(this->name, "{}Forward finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

auto Op::startAdjoint(CMap y, Map const &x, bool const ip) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Adjoint y [{}] expected [{}]", y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Adjoint x [{}] expected [{}]", x.rows(), cols()); }
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Adjoint [{},{}] |y| {}", (ip ? "IP " : ""), rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug(this->name, "{}Adjoint [{},{}]", (ip ? "IP " : ""), rows(), cols());
  }
  return Log::Now();
}

void Op::finishAdjoint(Map const &x, Log::Time const start, bool const ip) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "{}Adjoint finished in {} |x| {}", (ip ? "IP " : ""), Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "{}Adjoint finished in {}", (ip ? "IP " : ""), Log::ToNow(start));
  }
}

auto Op::startInverse(CMap y, Map const &x) const -> Log::Time
{
  if (y.rows() != rows()) { throw Log::Failure(this->name, "Inverse y [{}] expected [{}]", y.rows(), rows()); }
  if (x.rows() != cols()) { throw Log::Failure(this->name, "Inverse x [{}] expected [{}]", x.rows(), cols()); }
  if (Log::IsHigh()) {
    Log::Debug(this->name, "Inverse [{},{}] |y| {}", rows(), cols(), ParallelNorm(y));
  } else {
    Log::Debug(this->name, "Inverse [{},{}]", rows(), cols());
  }
  return Log::Now();
}

void Op::finishInverse(Map const &x, Log::Time const start) const
{
  if (Log::IsHigh()) {
    Log::Debug(this->name, "Inverse finished in {} |x| {}", Log::ToNow(start), ParallelNorm(x));
  } else {
    Log::Debug(this->name, "Inverse finished in {}", Log::ToNow(start));
  }
}

} // namespace rl::Ops
