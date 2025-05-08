#include "ops.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Ops {

template <typename S>
Identity<S>::Identity(Index const s)
  : Op<S>("Identity")
  , sz{s}
{
}

template <typename S> auto Identity<S>::rows() const -> Index { return sz; }

template <typename S> auto Identity<S>::cols() const -> Index { return sz; }

template <typename S> void Identity<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x;
  this->finishForward(y, time, false);
}

template <typename S> void Identity<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = y;
  this->finishAdjoint(x, time, false);
}

template <typename S> void Identity<S>::inverse(CMap const y, Map x) const
{
  auto const time = this->startInverse(y, x, false);
  x.device(Threads::CoreDevice())= y;
  this->finishInverse(x, time, false);
}

template <typename S> void Identity<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x;
  this->finishForward(y, time, true);
}

template <typename S> void Identity<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) += y;
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

template <typename S> void MatMul<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = mat * x;
  this->finishForward(y, time, false);
}

template <typename S> void MatMul<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = mat.adjoint() * y;
  this->finishAdjoint(x, time, false);
}

template <typename S> void MatMul<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += mat * x;
  this->finishForward(y, time, true);
}

template <typename S> void MatMul<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) += mat.adjoint() * y;
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

template <typename S> void DiagScale<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x * scale;
  this->finishForward(y, time, false);
}

template <typename S> void DiagScale<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = y * scale;
  this->finishAdjoint(x, time, false);
}

template <typename S> void DiagScale<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x * scale;
  this->finishForward(y, time, true);
}

template <typename S> void DiagScale<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) += y * scale;
  this->finishAdjoint(x, time, true);
}

template <typename S> auto DiagScale<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<DiagScale>(sz, 1.f / scale);
}

template struct DiagScale<float>;
template struct DiagScale<Cx>;

template <typename S>
DiagRep<S>::DiagRep(Vector const &v, Index const repI, Index const repO)
  : Op<S>("DiagRep")
  , s{v}
  , rI{repI}
  , rO{repO}
{
  Log::Debug("Op", "Diagonal Repeat. Weights min {} max {}", s.array().abs().minCoeff(), s.array().abs().maxCoeff());
}

template <typename S> auto DiagRep<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<DiagRep>(s.array().inverse(), rI, rO);
}

template <typename S> auto DiagRep<S>::rows() const -> Index { return s.rows() * rI * rO; }
template <typename S> auto DiagRep<S>::cols() const -> Index { return s.rows() * rI * rO; }

template <typename S> void DiagRep<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  auto const rep = s.array().transpose().replicate(rI, rO).reshaped();
  if (isInverse) {
    y.array().device(Threads::CoreDevice()) = x.array() / rep;
  } else {
    y.array().device(Threads::CoreDevice()) = x.array() * rep;
  }
  this->finishForward(y, time, false);
}

template <typename S> void DiagRep<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  auto const rep = s.array().transpose().replicate(rI, rO).reshaped();
  if (isInverse) {
    x.array().device(Threads::CoreDevice()) = y.array() / rep;
  } else {
    x.array().device(Threads::CoreDevice()) = y.array() * rep;
  }
  this->finishAdjoint(x, time, false);
}

template <typename S> void DiagRep<S>::inverse(CMap const y, Map x) const
{
  auto const time = this->startInverse(y, x, false);
  auto const rep = s.array().transpose().replicate(rI, rO).reshaped();
  if (isInverse) {
    x.array().device(Threads::CoreDevice()) = y.array() * rep;
  } else {
    x.array().device(Threads::CoreDevice()) = y.array() / rep;
  }
  this->finishInverse(x, time, false);
}

template <typename S> void DiagRep<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  auto const rep = s.array().transpose().replicate(rI, rO).reshaped();
  if (isInverse) {
    y.array().device(Threads::CoreDevice()) += x.array() / rep;
  } else {
    y.array().device(Threads::CoreDevice()) += x.array() * rep;
  }
  this->finishForward(y, time, true);
}

template <typename S> void DiagRep<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto       rep = s.array().transpose().replicate(rI, rO).reshaped();
  if (isInverse) {
    x.array().device(Threads::CoreDevice()) += y.array() / rep;
  } else {
    x.array().device(Threads::CoreDevice()) += y.array() * rep;
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
  , temp(B->rows())
{
  if (A->cols() != B->rows()) {
    throw Log::Failure("Op", "Multiply mismatched dimensions [{},{}] and [{},{}]", A->rows(), A->cols(), B->rows(), B->cols());
  }
}

template <typename S> auto Multiply<S>::inverse() const -> std::shared_ptr<Op<S>>
{
  return std::make_shared<Multiply<S>>(B->inverse(), A->inverse());
}

template <typename S> auto Multiply<S>::rows() const -> Index { return A->rows(); }
template <typename S> auto Multiply<S>::cols() const -> Index { return B->cols(); }

template <typename S> void Multiply<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->forward(tcm, y);
  this->finishForward(y, time, false);
}

template <typename S> void Multiply<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->adjoint(tcm, x);
  this->finishAdjoint(x, time, false);
}

template <typename S> void Multiply<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->iforward(tcm, y);
  this->finishForward(y, time, true);
}

template <typename S> void Multiply<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->iadjoint(tcm, x);
  this->finishAdjoint(x, time, true);
}

template struct Multiply<float>;
template struct Multiply<Cx>;

template <typename S> auto Mul(typename Op<S>::Ptr a, typename Op<S>::Ptr b) -> typename Op<S>::Ptr
{
  return std::make_shared<Ops::Multiply<S>>(a, b);
}

template auto Mul<float>(typename Op<float>::Ptr a, typename Op<float>::Ptr b) -> typename Op<float>::Ptr;
template auto Mul<Cx>(typename Op<Cx>::Ptr a, typename Op<Cx>::Ptr b) -> typename Op<Cx>::Ptr;


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
    if (ops[ii]->cols() != ops[ii + 1]->cols()) {
      throw Log::Failure("Op", "VStack {} {} had {} cols but op {} {} had {}", ii, ops[ii]->name, ops[ii]->cols(), ii + 1,
                         ops[ii + 1]->name, ops[ii + 1]->cols());
    }
  }
}

template <typename S> auto VStack<S>::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

template <typename S> auto VStack<S>::cols() const -> Index { return ops.front()->cols(); }

template <typename S> void VStack<S>::forward(CMap const x, Map y) const
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

template <typename S> void VStack<S>::adjoint(CMap const y, Map x) const
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

template <typename S> void VStack<S>::iforward(CMap const x, Map y) const
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

template <typename S> void VStack<S>::iadjoint(CMap const y, Map x) const
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
HStack<S>::HStack(std::vector<std::shared_ptr<Op<S>>> const &o)
  : Op<S>{"HStack"}
  , ops{o}
{
  check();
}

template <typename S>
HStack<S>::HStack(std::shared_ptr<Op<S>> op1, std::shared_ptr<Op<S>> op2)
  : Op<S>{"HStack"}
  , ops{op1, op2}
{
  check();
}

template <typename S>
HStack<S>::HStack(std::shared_ptr<Op<S>> op1, std::vector<std::shared_ptr<Op<S>>> const &others)
  : Op<S>{"HStack"}
  , ops{op1}
{
  ops.insert(ops.end(), others.begin(), others.end());
  check();
}

template <typename S> void HStack<S>::check()
{
  for (size_t ii = 0; ii < ops.size() - 1; ii++) {
    if (ops[ii]->rows() != ops[ii + 1]->rows()) {
      throw Log::Failure("Op", "HStack {} {} had {} rows but op {} {} had {}", ii, ops[ii]->name, ops[ii]->rows(), ii + 1,
                         ops[ii + 1]->name, ops[ii + 1]->rows());
    }
  }
}

template <typename S> auto HStack<S>::rows() const -> Index { return ops.front()->rows(); }

template <typename S> auto HStack<S>::cols() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->cols(); });
}

template <typename S> void HStack<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  Index      ic = 0;
  y.setZero();
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iforward(xm, y); // Need to sum, use in-place version
  }
  this->finishForward(y, time, false);
}

template <typename S> void HStack<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ic = 0;
  for (auto const &op : ops) {
    Map xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->adjoint(y, xm);
  }
  this->finishAdjoint(x, time, false);
}

template <typename S> void HStack<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  Index      ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iforward(xm, y); // Need to sum, use in-place version
  }
  this->finishForward(y, time, true);
}

template <typename S> void HStack<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ic = 0;
  for (auto const &op : ops) {
    Map xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iadjoint(y, xm);
  }
  this->finishAdjoint(x, time, true);
}

template struct HStack<float>;
template struct HStack<Cx>;

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

template <typename S> void DStack<S>::forward(CMap const x, Map y) const
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

template <typename S> void DStack<S>::adjoint(CMap const y, Map x) const
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

template <typename S> void DStack<S>::inverse(CMap const y, Map x) const
{
  auto const time = this->startInverse(y, x, false);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->inverse(ym, xm);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishInverse(x, time, false);
}

template <typename S> void DStack<S>::iforward(CMap const x, Map y) const
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

template <typename S> void DStack<S>::iadjoint(CMap const y, Map x) const
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

template <typename S> void Extract<S>::forward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x.segment(start, r);
  this->finishForward(y, time, false);
}

template <typename S> void Extract<S>::adjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.segment(0, start).setZero();
  x.segment(start, r).device(Threads::CoreDevice()) = y;
  x.segment(start + r, c - (start + r)).setZero();
  this->finishAdjoint(x, time, false);
}

template <typename S> void Extract<S>::iforward(CMap const x, Map y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x.segment(start, r);
  this->finishForward(y, time, true);
}

template <typename S> void Extract<S>::iadjoint(CMap const y, Map x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.segment(start, r).device(Threads::CoreDevice()) += y;
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
  if (a->rows() != b->rows()) {
    throw Log::Failure("Op", "Subtract operands must have same dimensions a [{},{}] b [{},{}]", //
                       a->rows(), a->cols(), b->rows(), b->cols());
  }
  if (a->rows() != b->rows()) {
    throw Log::Failure("Op", "Subtract operands must have same dimensions a [{},{}] b [{},{}]", //
                       a->rows(), a->cols(), b->rows(), b->cols());
  }
}

template <typename S> auto Subtract<S>::rows() const -> Index { return a->rows(); }
template <typename S> auto Subtract<S>::cols() const -> Index { return a->cols(); }

template <typename S> void Subtract<S>::forward(CMap const x, Map y) const
{ // Perform gymnastics to save memory
  auto const time = this->startForward(x, y, false);
  b->forward(x, y);
  y.device(Threads::CoreDevice()) = -y;
  a->iforward(x, y);
  this->finishForward(y, time, false);
}

template <typename S> void Subtract<S>::adjoint(CMap const y, Map x) const
{ // Perform gymnastics to save memory
  auto const time = this->startAdjoint(y, x, false);
  b->adjoint(y, x);
  x.device(Threads::CoreDevice()) = -x;
  a->iadjoint(y, x);
  this->finishAdjoint(x, time, false);
}

template <typename S> void Subtract<S>::iforward(CMap const x, Map y) const
{ // Perform gymnastics to save memory
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) = -y;
  b->iforward(x, y);
  y.device(Threads::CoreDevice()) = -y;
  a->iforward(x, y);
  this->finishForward(y, time, true);
}

template <typename S> void Subtract<S>::iadjoint(CMap const y, Map x) const
{ // Perform gymnastics to save memory
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) = -x;
  b->iadjoint(y, x);
  x.device(Threads::CoreDevice()) = -x;
  a->iadjoint(y, x);
  this->finishAdjoint(x, time, true);
}

template struct Subtract<float>;
template struct Subtract<Cx>;

template <typename S> auto Sub(typename Op<S>::Ptr a, typename Op<S>::Ptr b) -> typename Op<S>::Ptr
{
  return std::make_shared<Ops::Subtract<S>>(a, b);
}

template auto Sub<float>(typename Op<float>::Ptr a, typename Op<float>::Ptr b) -> typename Op<float>::Ptr;
template auto Sub<Cx>(typename Op<Cx>::Ptr a, typename Op<Cx>::Ptr b) -> typename Op<Cx>::Ptr;


} // namespace rl::Ops
