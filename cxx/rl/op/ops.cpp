#include "ops.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Ops {

Identity::Identity(Index const s)
  : Op("Identity")
  , sz{s}
{
}

auto Identity::Make(Index const s) -> Identity::Ptr { return std::make_shared<Identity>(s); }
auto Identity::rows() const -> Index { return sz; }
auto Identity::cols() const -> Index { return sz; }

void Identity::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x * s;
  this->finishForward(y, time, false);
}

void Identity::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = y * s;
  this->finishAdjoint(x, time, false);
}

void Identity::inverse(CMap y, Map x, float const s, float const b) const
{
  auto const time = this->startInverse(y, x);
  x.device(Threads::CoreDevice()) = y / (s + b);
  this->finishInverse(x, time);
}

void Identity::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x * s;
  this->finishForward(y, time, true);
}

void Identity::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) += y * s;
  this->finishAdjoint(x, time, true);
}

Adjoint::Adjoint(Ptr o)
  : Op("Adjoint")
  , op{o}
{
}

auto Adjoint::Make(Ptr o) -> Ptr { return std::make_shared<Adjoint>(o); }
auto Adjoint::rows() const -> Index { return op->cols(); }
auto Adjoint::cols() const -> Index { return op->rows(); }

void Adjoint::forward(CMap x, Map y, float const s) const { op->adjoint(x, y, s); }
void Adjoint::adjoint(CMap y, Map x, float const s) const { op->forward(y, x, s); }
void Adjoint::iforward(CMap x, Map y, float const s) const { op->iadjoint(x, y, s); }
void Adjoint::iadjoint(CMap y, Map x, float const s) const { op->iforward(y, x, s); }

MatMul::MatMul(Matrix const m)
  : Op("MatMul")
  , mat{m}
{
}

auto MatMul::rows() const -> Index { return mat.rows(); }

auto MatMul::cols() const -> Index { return mat.cols(); }

void MatMul::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = mat * x * s;
  this->finishForward(y, time, false);
}

void MatMul::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = mat.adjoint() * y * s;
  this->finishAdjoint(x, time, false);
}

void MatMul::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += mat * x * s;
  this->finishForward(y, time, true);
}

void MatMul::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) += mat.adjoint() * y * s;
  this->finishAdjoint(x, time, true);
}

DiagScale::DiagScale(Index const sz1, float const s1)
  : Op("DiagScale")
  , scale{s1}
  , sz{sz1}
{
}

auto DiagScale::Make(Index const sz, float const s) -> DiagScale::Ptr { return std::make_shared<DiagScale>(sz, s); }

auto DiagScale::rows() const -> Index { return sz; }
auto DiagScale::cols() const -> Index { return sz; }

void DiagScale::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x * scale * s;
  this->finishForward(y, time, false);
}

void DiagScale::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::CoreDevice()) = y * scale * s;
  this->finishAdjoint(x, time, false);
}

void DiagScale::inverse(CMap y, Map x, float const s, float const b) const
{
  auto const time = this->startInverse(y, x);
  x.device(Threads::CoreDevice()) = y / (scale * s + b);
  this->finishInverse(x, time);
}

void DiagScale::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x * scale * s;
  this->finishForward(y, time, true);
}

void DiagScale::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) += y * scale * s;
  this->finishAdjoint(x, time, true);
}

DiagRep::DiagRep(Vector const &d_, Index const repI, Index const repO)
  : Op("DiagRep")
  , d{d_}
  , rI{repI}
  , rO{repO}
{
  Log::Debug("Op", "Diagonal Repeat. Weights min {} max {}", d.array().abs().minCoeff(), d.array().abs().maxCoeff());
}

auto DiagRep::rows() const -> Index { return d.rows() * rI * rO; }
auto DiagRep::cols() const -> Index { return d.rows() * rI * rO; }

void DiagRep::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  auto const rep = d.array().transpose().replicate(rI, rO).reshaped();
  y.array().device(Threads::CoreDevice()) = x.array() * rep * s;
  this->finishForward(y, time, false);
}

void DiagRep::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  auto const rep = d.array().transpose().replicate(rI, rO).reshaped();
  x.array().device(Threads::CoreDevice()) = y.array() * rep * s;

  this->finishAdjoint(x, time, false);
}

void DiagRep::inverse(CMap y, Map x, float const s, float const b) const
{
  auto const time = this->startInverse(y, x);
  auto const rep = d.array().transpose().replicate(rI, rO).reshaped();
  x.array().device(Threads::CoreDevice()) = y.array() / (rep * s + b);
  this->finishInverse(x, time);
}

void DiagRep::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  auto const rep = d.array().transpose().replicate(rI, rO).reshaped();
  y.array().device(Threads::CoreDevice()) += x.array() * rep * s;
  this->finishForward(y, time, true);
}

void DiagRep::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto       rep = d.array().transpose().replicate(rI, rO).reshaped();
  x.array().device(Threads::CoreDevice()) += y.array() * rep * s;
  this->finishAdjoint(x, time, true);
}

Multiply::Multiply(Ptr AA, Ptr BB)
  : Op("Mult")
  , A{AA}
  , B{BB}
  , temp(B->rows())
{
  if (A->cols() != B->rows()) {
    throw Log::Failure("Op", "Multiply mismatched dimensions [{},{}] and [{},{}]", A->rows(), A->cols(), B->rows(), B->cols());
  }
}

auto Multiply::rows() const -> Index { return A->rows(); }
auto Multiply::cols() const -> Index { return B->cols(); }

void Multiply::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm, s);
  A->forward(tcm, y, s);
  this->finishForward(y, time, false);
}

void Multiply::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm, s);
  B->adjoint(tcm, x, s);
  this->finishAdjoint(x, time, false);
}

void Multiply::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  B->forward(x, tm);
  A->iforward(tcm, y, s);
  this->finishForward(y, time, true);
}

void Multiply::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Map        tm(temp.data(), temp.size());
  CMap       tcm(temp.data(), temp.size());
  A->adjoint(y, tm);
  B->iadjoint(tcm, x, s);
  this->finishAdjoint(x, time, true);
}

auto Mul(typename Op::Ptr a, typename Op::Ptr b) -> typename Op::Ptr { return std::make_shared<Ops::Multiply>(a, b); }

VStack::VStack(std::vector<Ptr> const &o)
  : Op{"VStack"}
  , ops{o}
{
  check();
}

VStack::VStack(Ptr op1, std::vector<Ptr> const &others)
  : Op{"VStack"}
  , ops{op1}
{
  ops.insert(ops.end(), others.begin(), others.end());
  check();
}

auto VStack::Make(std::vector<Ptr> const &o) -> Ptr { return std::make_shared<VStack>(o); }
auto VStack::Make(Ptr o1, std::vector<Ptr> const &o) -> Ptr { return std::make_shared<VStack>(o1, o); }

void VStack::check()
{
  for (size_t ii = 0; ii < ops.size() - 1; ii++) {
    if (ops[ii]->cols() != ops[ii + 1]->cols()) {
      throw Log::Failure("Op", "VStack {} {} had {} cols but op {} {} had {}", ii, ops[ii]->name, ops[ii]->cols(), ii + 1,
                         ops[ii + 1]->name, ops[ii + 1]->cols());
    }
  }
}

auto VStack::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

auto VStack::cols() const -> Index { return ops.front()->cols(); }

void VStack::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  Index      ir = 0;
  for (auto const &op : ops) {
    Map ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->forward(x, ym, s);
  }
  this->finishForward(y, time, false);
}

void VStack::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  Map        xtm(x.data(), x.rows());
  x.setConstant(0.f);
  Index ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iadjoint(ym, xtm, s); // Need to sum, use the in-place version
  }
  this->finishAdjoint(x, time, false);
}

void VStack::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Index      ir = 0;
  for (auto const &op : ops) {
    Map ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iforward(x, ym, s);
  }
  this->finishForward(y, time, true);
}

void VStack::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Map        xtm(x.data(), x.rows());
  Index      ir = 0;
  for (auto const &op : ops) {
    CMap ym(y.data() + ir, op->rows());
    ir += op->rows();
    op->iadjoint(ym, xtm, s);
  }
  this->finishAdjoint(x, time, true);
}

HStack::HStack(std::vector<Ptr> const &o)
  : Op{"HStack"}
  , ops{o}
{
  check();
}

HStack::HStack(Ptr op1, Ptr op2)
  : Op{"HStack"}
  , ops{op1, op2}
{
  check();
}

HStack::HStack(Ptr op1, std::vector<Ptr> const &others)
  : Op{"HStack"}
  , ops{op1}
{
  ops.insert(ops.end(), others.begin(), others.end());
  check();
}

void HStack::check()
{
  for (size_t ii = 0; ii < ops.size() - 1; ii++) {
    if (ops[ii]->rows() != ops[ii + 1]->rows()) {
      throw Log::Failure("Op", "HStack {} {} had {} rows but op {} {} had {}", ii, ops[ii]->name, ops[ii]->rows(), ii + 1,
                         ops[ii + 1]->name, ops[ii + 1]->rows());
    }
  }
}

auto HStack::rows() const -> Index { return ops.front()->rows(); }

auto HStack::cols() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->cols(); });
}

void HStack::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  Index      ic = 0;
  y.setZero();
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iforward(xm, y, s); // Need to sum, use in-place version
  }
  this->finishForward(y, time, false);
}

void HStack::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ic = 0;
  for (auto const &op : ops) {
    Map xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->adjoint(y, xm, s);
  }
  this->finishAdjoint(x, time, false);
}

void HStack::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Index      ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iforward(xm, y, s); // Need to sum, use in-place version
  }
  this->finishForward(y, time, true);
}

void HStack::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ic = 0;
  for (auto const &op : ops) {
    Map xm(x.data() + ic, op->cols());
    ic += op->cols();
    op->iadjoint(y, xm, s);
  }
  this->finishAdjoint(x, time, true);
}

DStack::DStack(std::vector<Ptr> const &o)
  : Op{"DStack"}
  , ops{o}
{
}

DStack::DStack(Ptr op1, Ptr op2)
  : Op{"DStack"}
  , ops{op1, op2}
{
}

auto DStack::rows() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->rows(); });
}

auto DStack::cols() const -> Index
{
  return std::accumulate(this->ops.begin(), this->ops.end(), 0L, [](Index a, auto const &op) { return a + op->cols(); });
}

void DStack::forward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    Map  ym(y.data() + ir, op->rows());
    op->forward(xm, ym, s);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishForward(y, time, false);
}

void DStack::adjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->adjoint(ym, xm, s);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishAdjoint(x, time, false);
}

void DStack::inverse(CMap y, Map x, float const s, float const b) const
{
  auto const time = this->startInverse(y, x);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->inverse(ym, xm, s, b);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishInverse(x, time);
}

void DStack::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    CMap xm(x.data() + ic, op->cols());
    Map  ym(y.data() + ir, op->rows());
    op->iforward(xm, ym, s);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishForward(y, time, true);
}

void DStack::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Index      ir = 0, ic = 0;
  for (auto const &op : ops) {
    Map  xm(x.data() + ic, op->cols());
    CMap ym(y.data() + ir, op->rows());
    op->iadjoint(ym, xm, s);
    ir += op->rows();
    ic += op->cols();
  }
  assert(ir == rows());
  assert(ic == cols());
  this->finishAdjoint(x, time, true);
}

Extract::Extract(Index const cols, Index const st, Index const rows)
  : Op("Extrct")
  , r{rows}
  , c{cols}
  , start{st}
{
}

auto Extract::rows() const -> Index { return r; }
auto Extract::cols() const -> Index { return c; }

void Extract::forward(CMap x, Map y, float const) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::CoreDevice()) = x.segment(start, r);
  this->finishForward(y, time, false);
}

void Extract::adjoint(CMap y, Map x, float const) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.segment(0, start).setZero();
  x.segment(start, r).device(Threads::CoreDevice()) = y;
  x.segment(start + r, c - (start + r)).setZero();
  this->finishAdjoint(x, time, false);
}

void Extract::iforward(CMap x, Map y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) += x.segment(start, r) * s;
  this->finishForward(y, time, true);
}

void Extract::iadjoint(CMap y, Map x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.segment(start, r).device(Threads::CoreDevice()) += y * s;
  this->finishAdjoint(x, time, true);
}

Subtract::Subtract(Ptr aa, Ptr bb)
  : Op("Sub")
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

auto Subtract::rows() const -> Index { return a->rows(); }
auto Subtract::cols() const -> Index { return a->cols(); }

void Subtract::forward(CMap x, Map y, float const s) const
{ // Perform gymnastics to save memory
  auto const time = this->startForward(x, y, false);
  b->forward(x, y, s);
  y.device(Threads::CoreDevice()) = -y;
  a->iforward(x, y, s);
  this->finishForward(y, time, false);
}

void Subtract::adjoint(CMap y, Map x, float const s) const
{ // Perform gymnastics to save memory
  auto const time = this->startAdjoint(y, x, false);
  b->adjoint(y, x, s);
  x.device(Threads::CoreDevice()) = -x;
  a->iadjoint(y, x, s);
  this->finishAdjoint(x, time, false);
}

void Subtract::iforward(CMap x, Map y, float const s) const
{ // Perform gymnastics to save memory
  auto const time = this->startForward(x, y, true);
  y.device(Threads::CoreDevice()) = -y;
  b->iforward(x, y, s);
  y.device(Threads::CoreDevice()) = -y;
  a->iforward(x, y, s);
  this->finishForward(y, time, true);
}

void Subtract::iadjoint(CMap y, Map x, float const s) const
{ // Perform gymnastics to save memory
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::CoreDevice()) = -x;
  b->iadjoint(y, x, s);
  x.device(Threads::CoreDevice()) = -x;
  a->iadjoint(y, x, s);
  this->finishAdjoint(x, time, true);
}

auto Sub(typename Op::Ptr a, typename Op::Ptr b) -> typename Op::Ptr { return std::make_shared<Ops::Subtract>(a, b); }

} // namespace rl::Ops
