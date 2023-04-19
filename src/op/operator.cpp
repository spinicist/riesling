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
  assert(x.rows() == cols());
  Vector y(this->rows());
  Map ym(y.data(), y.size());
  this->forward(CMap(x.data(), x.size()), ym);
  return y;
}

template <typename S>
auto Op<S>::adjoint(Vector const &y) const -> Vector
{
  Log::Print<Log::Level::Debug>("Op {} adjoint y {} rows {} cols {}", name, y.rows(), rows(), cols());
  assert(y.rows() == rows());
  Vector x(this->cols());
  Map xm(x.data(), x.size());
  this->adjoint(CMap(y.data(), y.size()), xm);
  return x;
}

template <typename S>
void Op<S>::forward(Vector const &x, Vector &y) const
{
  Log::Print<Log::Level::Debug>("Op {} forward x {} y {} rows {} cols {}", name, x.rows(), y.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  CMap xm(x.data(), x.size());
  Map ym(y.data(), y.size());
  this->forward(xm, ym);
}

template <typename S>
void Op<S>::adjoint(Vector const &y, Vector &x) const
{
  Log::Print<Log::Level::Debug>("Op {} adjoint y {} x {} rows {} cols {}", name, y.rows(), x.rows(), rows(), cols());
  assert(x.rows() == cols());
  assert(y.rows() == rows());
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
  assert(x.rows() == y.rows() && x.rows() == sz);
  y = x;
}

template <typename S>
void Identity<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == y.rows() && x.rows() == sz);
  x = y;
}

template struct Identity<float>;
template struct Identity<Cx>;

template <typename S>
Scale<S>::Scale(std::shared_ptr<Op<S>> o, float const s)
  : Op<S>("Scale")
  , op{o}
  , scale{s}
{
}

template <typename S>
auto Scale<S>::rows() const -> Index
{
  return op->rows();
}
template <typename S>
auto Scale<S>::cols() const -> Index
{
  return op->cols();
}

template <typename S>
void Scale<S>::forward(CMap const &x, Map &y) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  op->forward(x, y);
  y *= scale;
}

template <typename S>
void Scale<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  op->adjoint(y, x);
  x *= scale;
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
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Vector temp(a->cols());
  Map tm(temp.data(), temp.size());
  CMap tcm(temp.data(), temp.size());
  a->forward(x, tm);
  b->forward(tcm, y);
}

template <typename S>
void Concat<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
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
  check();
}

template <typename S>
VStack<S>::VStack(std::shared_ptr<Op<S>> op1, std::shared_ptr<Op<S>> op2)
  : Op<S>{"VStack"}
  , ops{op1, op2}
{
  check();
}

template <typename S>
void VStack<S>::check()
{
  for (auto ii = 0; ii < ops.size() - 1; ii++) {
    if (ops[ii]->cols() != ops[ii + 1]->cols()) {
      Log::Fail(
        "Operator {} had {} columns, {} had {}", ops[ii]->name, ops[ii]->cols(), ops[ii + 1]->name, ops[ii + 1]->cols());
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
  assert(x.rows() == cols());
  assert(y.rows() == rows());
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
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Vector xt(x.rows());
  Map xtm(xt.data(), xt.rows());
  x.setConstant(0.f);
  Index ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->adjoint(ym, xtm);
    x += xt;
  }
}

template struct VStack<float>;
template struct VStack<Cx>;

template <typename S>
DStack<S>::DStack(std::vector<std::shared_ptr<Op<S>>> const &o)
  : Op<S>{"DStack"}
  , ops{o}
{
}

template <typename S>
DStack<S>::DStack(std::shared_ptr<Op<S>> op1, std::shared_ptr<Op<S>> op2)
  : Op<S>{"DStack"}
  , ops{op1, op2}
{
}

template <typename S>
auto DStack<S>::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

template <typename S>
auto DStack<S>::cols() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->cols(); });
}

template <typename S>
void DStack<S>::forward(CMap const &x, Map &y) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Index ir = 0, ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    Map ym(y.data() + ir, op->rows());
    op->forward(xm, ym);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir = rows());
  assert(ic = cols());
}

template <typename S>
void DStack<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Index ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->adjoint(ym, xm);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir = rows());
  assert(ic = cols());
}

template struct DStack<float>;
template struct DStack<Cx>;

} // namespace LinOps

} // namespace rl
