#include "operator.hpp"

#include <numeric>

namespace rl {

namespace LinOps {

template <typename S>
Op<S>::Op(std::string const &n)
  : name{n}
{
}

template <typename S>
auto Op<S>::forward(Vector const &x) const -> Vector
{
  Log::Print<Log::Level::Debug>("Op {} forward x {} rows {} cols {}", name, x.rows(), rows(), cols());
  Vector y(this->rows());
  Map ym(y.data(), y.size());
  this->forward(CMap(x.data(), x.size()), ym);
  return y;
}

template <typename S>
auto Op<S>::adjoint(Vector const &y) const -> Vector
{
  Log::Print<Log::Level::Debug>("Op {} adjoint y {} rows {} cols {}", name, y.rows(), rows(), cols());
  Vector x(this->cols());
  Map xm(x.data(), x.size());
  this->adjoint(CMap(y.data(), y.size()), xm);
  return x;
}

template <typename S>
void Op<S>::forward(Vector const &x, Vector &y) const
{
  Log::Print<Log::Level::Debug>("Op {} forward x {} y {} rows {} cols {}", name, x.rows(), y.rows(), rows(), cols());
  CMap xm(x.data(), x.size());
  Map ym(y.data(), y.size());
  this->forward(xm, ym);
}

template <typename S>
void Op<S>::adjoint(Vector const &y, Vector &x) const
{
  Log::Print<Log::Level::Debug>("Op {} adjoint y {} x {} rows {} cols {}", name, y.rows(), x.rows(), rows(), cols());
  CMap ym(y.data(), y.size());
  Map xm(x.data(), x.size());
  this->adjoint(ym, xm);
}

template struct Op<float>;
template struct Op<Cx>;

template <typename S>
Identity<S>::Identity(Index const s)
  : Op<S>("Identity")
  , sz{s}
{
}

template <typename S>
auto Identity<S>::rows() const -> Index
{
  return sz;
}

template <typename S>
auto Identity<S>::cols() const -> Index
{
  return sz;
}

template <typename S>
void Identity<S>::forward(CMap const &x, Map &y) const
{
  y = x;
}

template <typename S>
void Identity<S>::adjoint(CMap const &y, Map &x) const
{
  x = y;
}

template struct Identity<float>;
template struct Identity<Cx>;

template <typename S>
Scale<S>::Scale(Index const size, float const s)
  : Op<S>("Scale")
  , sz{size}
  , scale{s}
{
}

template <typename S>
auto Scale<S>::rows() const -> Index
{
  return sz;
}
template <typename S>
auto Scale<S>::cols() const -> Index
{
  return sz;
}

template <typename S>
void Scale<S>::forward(CMap const &x, Map &y) const
{
  y = x * scale;
}
template <typename S>
void Scale<S>::adjoint(CMap const &y, Map &x) const
{
  x = y * scale;
}

template struct Scale<float>;
template struct Scale<Cx>;

template <typename S>
Concat<S>::Concat(std::shared_ptr<Op<S>> a_, std::shared_ptr<Op<S>> b_)
  : Op<S>("Concat")
  , a{a_}
  , b{b_}
{
}

template <typename S>
auto Concat<S>::rows() const -> Index
{
  return a->rows();
}
template <typename S>
auto Concat<S>::cols() const -> Index
{
  return b->cols();
}

template <typename S>
void Concat<S>::forward(CMap const &x, Map &y) const
{
  Vector temp(a->cols());
  Map tm(temp.data(), temp.size());
  CMap tcm(temp.data(), temp.size());
  a->forward(x, tm);
  b->forward(tcm, y);
}

template <typename S>
void Concat<S>::adjoint(CMap const &y, Map &x) const
{
  Vector temp(b->cols());
  Map tm(temp.data(), temp.size());
  CMap tcm(temp.data(), temp.size());
  b->adjoint(y, tm);
  a->adjoint(tcm, x);
}

template struct Concat<float>;
template struct Concat<Cx>;

template <typename S>
VStack<S>::VStack(std::vector<std::shared_ptr<Op<S>>> const &o)
  : Op<S>{"VStack"}
  , ops{o}
{
  for (auto ii = 1; ii < ops.size(); ii++) {
    if (ops[ii]->cols() != ops[ii - 1]->cols()) {
      Log::Fail("Operators had mismatched number of columns");
    }
  }
}

template <typename S>
auto VStack<S>::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

template <typename S>
auto VStack<S>::cols() const -> Index
{
  return ops.front()->cols();
}

template <typename S>
void VStack<S>::forward(CMap const &x, Map &y) const
{
  Index ir = 0;
  for (auto const &op : ops) {
    Map ym(y.data() + ir, op->rows());
    op->forward(x, ym);
    ir += op->rows();
  }
}

template <typename S>
void VStack<S>::adjoint(CMap const &y, Map &x) const
{
  Vector xt(x.size());
  Map xtm(xt.data(), xt.size());
  x.setConstant(0.f);
  Index ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    op->adjoint(ym, xtm);
    x += xt;
    ir += op->rows();
  }
}

template struct VStack<float>;
template struct VStack<Cx>;

} // namespace Op

} // namespace rl
