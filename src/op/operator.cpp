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
Multiply<S>::Multiply(std::shared_ptr<Op<S>> AA, std::shared_ptr<Op<S>> BB)
  : Op<S>("Multiply")
  , A{AA}
  , B{BB}
{
  assert(A->cols() == B->rows());
}

template <typename S>
auto Multiply<S>::rows() const -> Index
{
  return A->rows();
}
template <typename S>
auto Multiply<S>::cols() const -> Index
{
  return B->cols();
}

template <typename S>
void Multiply<S>::forward(CMap const &x, Map &y) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Vector temp(B->rows());
  Map tm(temp.data(), temp.size());
  CMap tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->forward(tcm, y);
}

template <typename S>
void Multiply<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  Vector temp(A->cols());
  Map tm(temp.data(), temp.size());
  CMap tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->adjoint(tcm, x);
}

template struct Multiply<float>;
template struct Multiply<Cx>;

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
    assert(ops[ii]->cols() == ops[ii + 1]->cols());
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

template <typename S>
Extract<S>::Extract(Index const cols, Index const st, Index const rows)
  : Op<S>("Extract")
  , r{rows}
  , c{cols}
  , start{st}
{
}

template <typename S>
auto Extract<S>::rows() const -> Index
{
  return r;
}
template <typename S>
auto Extract<S>::cols() const -> Index
{
  return c;
}

template <typename S>
void Extract<S>::forward(CMap const &x, Map &y) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  y = x.segment(start, r);
}

template <typename S>
void Extract<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  x.segment(0, start).setZero();
  x.segment(start, r) = y;
  x.segment(start + r, c - (start + r)).setZero();
}

template struct Extract<float>;
template struct Extract<Cx>;

template <typename S>
Subtract<S>::Subtract(std::shared_ptr<Op<S>> aa, std::shared_ptr<Op<S>> bb)
  : Op<S>("Subtract")
  , a{aa}
  , b{bb}
{
  assert(a->rows() == b->rows());
  assert(a->cols() == b->cols());
}

template <typename S>
auto Subtract<S>::rows() const -> Index
{
  return a->rows();
}
template <typename S>
auto Subtract<S>::cols() const -> Index
{
  return a->cols();
}

template <typename S>
void Subtract<S>::forward(CMap const &x, Map &y) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  a->forward(x, y);
  Vector temp(rows());
  Map tm(temp.data(), temp.rows());
  b->forward(x, tm);
  y -= tm;
}

template <typename S>
void Subtract<S>::adjoint(CMap const &y, Map &x) const
{
  assert(x.rows() == cols());
  assert(y.rows() == rows());
  a->adjoint(y, x);
  Vector temp(cols());
  Map tm(temp.data(), temp.rows());
  b->adjoint(y, tm);
  x -= tm;
}

template struct Subtract<float>;
template struct Subtract<Cx>;

} // namespace LinOps

} // namespace rl
