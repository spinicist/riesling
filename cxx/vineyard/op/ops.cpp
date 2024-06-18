#include "ops.hpp"

namespace rl::Ops {

template <typename S>
Identity<S>::Identity(Index const s)
  : Op<S>("Identity")
  , sz{s}
{
}

template <typename S> auto Identity<S>::rows() const -> Index { return sz; }

template <typename S> auto Identity<S>::cols() const -> Index { return sz; }

template <typename S> void Identity<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  y = x;
  this->finishForward(y, time, false);
}

template <typename S> void Identity<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x = y;
  this->finishAdjoint(x, time, false);
}

template <typename S> void Identity<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  y += x;
  this->finishForward(y, time, true);
}

template <typename S> void Identity<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x += y;
  this->finishAdjoint(x, time, true);
}

template struct Identity<float>;
template struct Identity<Cx>;

template <typename S>
MatMul<S>::MatMul(Matrix const m)
  : Op<S>("MatMul")
  , mat{m}
{
}

template <typename S> auto MatMul<S>::rows() const -> Index { return mat.rows(); }

template <typename S> auto MatMul<S>::cols() const -> Index { return mat.cols(); }

template <typename S> void MatMul<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  y = mat * x;
  this->finishForward(y, time, false);
}

template <typename S> void MatMul<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x = mat.adjoint() * y;
  this->finishAdjoint(x, time, false);
}

template <typename S> void MatMul<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  y += mat * x;
  this->finishForward(y, time, true);
}

template <typename S> void MatMul<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x += mat.adjoint() * y;
  this->finishAdjoint(x, time, true);
}

template struct MatMul<float>;
template struct MatMul<Cx>;

template <typename S>
DiagScale<S>::DiagScale(Index const sz1, float const s1)
  : Op<S>("DiagScale")
  , scale{s1}
  , sz{sz1}
{
}

template <typename S> auto DiagScale<S>::rows() const -> Index { return sz; }
template <typename S> auto DiagScale<S>::cols() const -> Index { return sz; }

template <typename S> void DiagScale<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  y = x * scale;
  this->finishForward(y, time, false);
}

template <typename S> void DiagScale<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x = y * scale;
  this->finishAdjoint(x, time, false);
}

template <typename S> void DiagScale<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  y += x * scale;
  this->finishForward(y, time, true);
}

template <typename S> void DiagScale<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x += y * scale;
  this->finishAdjoint(x, time, true);
}

template <typename S> auto DiagScale<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<DiagScale>(sz, 1.f / scale);
}

template struct DiagScale<float>;
template struct DiagScale<Cx>;

template <typename S>
DiagRep<S>::DiagRep(Index const n, Vector const &v)
  : Op<S>("DiagRep")
  , reps{n}
  , s{v}
{
  Log::Print("Weights min {} max {}", s.array().abs().minCoeff(), s.array().abs().maxCoeff());
}

template <typename S>
DiagRep<S>::DiagRep(Index const n, Vector const &v, float const b, float const sc)
  : Op<S>("DiagRep")
  , reps{n}
  , s{v}
  , isInverse{true}
  , bias{b}
  , scale{sc}
{
}

template <typename S> auto DiagRep<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<DiagRep>(reps, s.array().inverse());
}

template <typename S> auto DiagRep<S>::inverse(float const b, float const sc) const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<DiagRep>(reps, s.array(), b, sc);
}

template <typename S> auto DiagRep<S>::rows() const -> Index { return s.rows() * reps; }
template <typename S> auto DiagRep<S>::cols() const -> Index { return s.rows() * reps; }

template <typename S> void DiagRep<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  auto       rep = s.array().transpose().replicate(reps, 1).reshaped();
  if (isInverse) {
    y.array() = x.array() / (bias + scale * rep);
  } else {
    y.array() = x.array() * rep;
  }
  this->finishForward(y, time, false);
}

template <typename S> void DiagRep<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  auto       rep = s.array().transpose().replicate(reps, 1).reshaped();
  if (isInverse) {
    x.array() = y.array() / (bias + scale * rep);
  } else {
    x.array() = y.array() * rep;
  }
  this->finishAdjoint(x, time, false);
}

template <typename S> void DiagRep<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  auto       rep = s.array().transpose().replicate(reps, 1).reshaped();
  if (isInverse) {
    y.array() += x.array() / (bias + scale * rep);
  } else {
    y.array() += x.array() * rep;
  }
  this->finishForward(y, time, true);
}

template <typename S> void DiagRep<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto       rep = s.array().transpose().replicate(reps, 1).reshaped();
  if (isInverse) {
    x.array() += y.array() / (bias + scale * rep);
  } else {
    x.array() += y.array() * rep;
  }
  this->finishAdjoint(x, time, true);
}

template struct DiagRep<float>;
template struct DiagRep<Cx>;

template <typename S>
Multiply<S>::Multiply(std::shared_ptr<Op<S>> AA, std::shared_ptr<Op<S>> BB)
  : Op<S>("Multiply")
  , A{AA}
  , B{BB}
{
  if (A->cols() != B->rows()) {
    Log::Fail("Multiply Op mismatched dimensions [{},{}] and [{},{}]", A->rows(), A->cols(), B->rows(), B->cols());
  }
}

template <typename S> auto Multiply<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<Multiply<S>>(B->inverse(), A->inverse());
}

template <typename S> auto Multiply<S>::rows() const -> Index { return A->rows(); }
template <typename S> auto Multiply<S>::cols() const -> Index { return B->cols(); }

template <typename S> void Multiply<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  Vector     temp(B->rows());
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->forward(tcm, y);
  this->finishForward(y, time, false);
}

template <typename S> void Multiply<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Vector     temp(A->cols());
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->adjoint(tcm, x);
  this->finishAdjoint(x, time, false);
}

template <typename S> void Multiply<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  Vector     temp(B->rows());
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->iforward(tcm, y);
  this->finishForward(y, time, true);
}

template <typename S> void Multiply<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Vector     temp(A->cols());
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->iadjoint(tcm, x);
  this->finishAdjoint(x, time, true);
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
VStack<S>::VStack(std::shared_ptr<Op<S>> op1, std::vector<std::shared_ptr<Op<S>>> const &others)
  : Op<S>{"VStack"}
  , ops{op1}
{
  ops.insert(ops.end(), others.begin(), others.end());
  check();
}

template <typename S> void VStack<S>::check()
{
  for (size_t ii = 0; ii < ops.size() - 1; ii++) {
    assert(ops[ii]->cols() == ops[ii + 1]->cols());
  }
}

template <typename S> auto VStack<S>::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

template <typename S> auto VStack<S>::cols() const -> Index { return ops.front()->cols(); }

template <typename S> void VStack<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  Index      ir = 0;
  for (auto const &op : ops) {
    Map ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->forward(x, ym);
  }
  this->finishForward(y, time, false);
}

template <typename S> void VStack<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Map        xtm(x.data(), x.rows());
  x.setConstant(0.f);
  Index ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iadjoint(ym, xtm); // Need to sum, use the in-place version
  }
  this->finishAdjoint(x, time, false);
}

template <typename S> void VStack<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  Index      ir = 0;
  for (auto const &op : ops) {
    Map ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iforward(x, ym);
  }
  this->finishForward(y, time, true);
}

template <typename S> void VStack<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Map        xtm(x.data(), x.rows());
  Index      ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iadjoint(ym, xtm);
  }
  this->finishAdjoint(x, time, true);
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

template <typename S> auto DStack<S>::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

template <typename S> auto DStack<S>::cols() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->cols(); });
}

template <typename S> void DStack<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    Map  ym(y.data() + ir, op->rows());
    op->forward(xm, ym);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishForward(y, time, false);
}

template <typename S> void DStack<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->adjoint(ym, xm);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishAdjoint(x, time, false);
}

template <typename S> void DStack<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    Map  ym(y.data() + ir, op->rows());
    op->iforward(xm, ym);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishForward(y, time, true);
}

template <typename S> void DStack<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->iadjoint(ym, xm);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishAdjoint(x, time, true);
}

template <typename S> auto DStack<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  std::vector<std::shared_ptr<Op<S>>> inverses(ops.size());
  for (size_t ii = 0; ii < ops.size(); ii++) {
    inverses[ii] = ops[ii]->inverse();
  }
  return std::make_shared<DStack<S>>(inverses);
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

template <typename S> auto Extract<S>::rows() const -> Index { return r; }
template <typename S> auto Extract<S>::cols() const -> Index { return c; }

template <typename S> void Extract<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  y = x.segment(start, r);
  this->finishForward(y, time, false);
}

template <typename S> void Extract<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.segment(0, start).setZero();
  x.segment(start, r) = y;
  x.segment(start + r, c - (start + r)).setZero();
  this->finishAdjoint(x, time, false);
}

template <typename S> void Extract<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  y += x.segment(start, r);
  this->finishForward(y, time, true);
}

template <typename S> void Extract<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.segment(start, r) += y;
  this->finishAdjoint(x, time, true);
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

template <typename S> auto Subtract<S>::rows() const -> Index { return a->rows(); }
template <typename S> auto Subtract<S>::cols() const -> Index { return a->cols(); }

template <typename S> void Subtract<S>::forward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, false);
  a->forward(x, y);
  Vector temp(rows());
  Map    tm(temp.data(), temp.rows());
  b->forward(x, tm);
  y -= tm;
  this->finishForward(y, time, false);
}

template <typename S> void Subtract<S>::adjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  a->adjoint(y, x);
  Vector temp(cols());
  Map    tm(temp.data(), temp.rows());
  b->adjoint(y, tm);
  x -= tm;
  this->finishAdjoint(x, time, false);
}

template <typename S> void Subtract<S>::iforward(CMap const &x, Map &y) const
{
  auto const time = this->startForward(x, y, true);
  a->iforward(x, y);
  Vector temp(rows());
  Map    tm(temp.data(), temp.rows());
  b->forward(x, tm);
  y -= tm;
  this->finishForward(y, time, true);
}

template <typename S> void Subtract<S>::iadjoint(CMap const &y, Map &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  a->iadjoint(y, x);
  Vector temp(cols());
  Map    tm(temp.data(), temp.rows());
  b->adjoint(y, tm);
  x -= tm;
  this->finishAdjoint(x, time, true);
}

template struct Subtract<float>;
template struct Subtract<Cx>;

} // namespace rl::Ops
