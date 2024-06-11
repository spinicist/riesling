#include "op.hpp"

namespace rl::Ops {

template <typename S>
Op<S>::Op(std::string const &n)
  : name{n}
{
}

template <typename S> void Op<S>::forward(Vector const &x, Vector &y) const
{
  Log::Debug("Op {} forward x {} y {} rows {} cols {}", name, x.rows(), y.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->forward(xm, ym);
}

template <typename S> void Op<S>::adjoint(Vector const &y, Vector &x) const
{
  Log::Debug("Op {} adjoint y {} x {} rows {} cols {}", name, y.rows(), x.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->adjoint(ym, xm);
}

template <typename S> auto Op<S>::forward(Vector const &x) const -> Vector
{
  Log::Debug("Op {} forward x {} rows {} cols {}", name, x.rows(), rows(), cols());
  assert(x.rows() == cols());
  Vector y(this->rows());
  Map    ym(y.data(), y.size());
  y.setZero();
  this->forward(CMap(x.data(), x.size()), ym);
  return y;
}

template <typename S> auto Op<S>::adjoint(Vector const &y) const -> Vector
{
  Log::Debug("Op {} adjoint y {} rows {} cols {}", name, y.rows(), rows(), cols());
  assert(y.rows() == rows());
  Vector x(this->cols());
  Map    xm(x.data(), x.size());
  x.setZero();
  this->adjoint(CMap(y.data(), y.size()), xm);
  return x;
}

template <typename S> void Op<S>::iforward(Vector const &x, Vector &y) const
{
  Log::Debug("Op {} iforward x {} y {} rows {} cols {}", name, x.rows(), y.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  CMap xm(x.data(), x.size());
  Map  ym(y.data(), y.size());
  this->iforward(xm, ym);
}

template <typename S> void Op<S>::iadjoint(Vector const &y, Vector &x) const
{
  Log::Debug("Op {} iadjoint y {} x {} rows {} cols {}", name, y.rows(), x.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  CMap ym(y.data(), y.size());
  Map  xm(x.data(), x.size());
  this->iadjoint(ym, xm);
}

template <typename S> auto Op<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  Log::Fail("Op {} does not have an inverse", name);
}

template <typename S> auto Op<S>::inverse(float const bias, float const scale) const -> std::shared_ptr<Op<S>>
{
  Log::Fail("Op {} does not have an inverse", name);
}

template <typename S> auto Op<S>::operator+(S const) const -> std::shared_ptr<Op<S>>
{
  Log::Fail("Op {} does not have operator+", name);
}

template struct Op<float>;
template struct Op<Cx>;

} // namespace rl::Ops
