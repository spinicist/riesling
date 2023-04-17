#pragma once

#include "log.hpp"
#include "types.hpp"

namespace rl {

namespace Op {

template <typename Scalar_ = Cx>
struct Operator
{
  using Scalar = Scalar_;
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;

  std::string name;

  Operator(std::string const &n)
    : name{n}
  {
  }

  virtual auto rows() const -> Index = 0;
  virtual auto cols() const -> Index = 0;
  virtual void forward(CMap const &x, Map &y) const = 0;
  virtual void adjoint(CMap const &x, Map &y) const = 0;

  auto forward(Vector const &x) const -> Vector
  {
    Vector y(this->rows());
    Map ym(y.data(), y.size());
    this->forward(CMap(x.data(), x.size()), ym);
    return y;
  }

  auto adjoint(Vector const &y) const -> Vector
  {
    Vector x(this->cols());
    Map xm(x.data(), x.size());
    this->adjoint(CMap(y.data(), y.size()), xm);
    return x;
  }
};

template <typename Scalar = Cx>
struct Identity final : Operator<Scalar>
{
  using typename Operator<Scalar>::Map;
  using typename Operator<Scalar>::CMap;

  Identity(Index const s)
    : sz{s}
  {
  }

  auto rows() const -> Index { return sz; }
  auto cols() const -> Index { return sz; }

  void forward(CMap const &x, Map &y) const { y = x; }
  void adjoint(CMap const &y, Map &x) const { x = y; }

private:
  Index sz;
};

template <typename Scalar = Cx>
struct Scale final : Operator<Scalar>
{
  using typename Operator<Scalar>::Map;
  using typename Operator<Scalar>::CMap;

  Scale(Index const size, float const s)
    : sz{size}, scale{s}
  {
  }

  auto rows() const -> Index { return sz; }
  auto cols() const -> Index { return sz; }

  void forward(CMap const &x, Map &y) const { y = x * scale; }
  void adjoint(CMap const &y, Map &x) const { x = y * scale; }

  Index sz;
  float scale;
};

template <typename Scalar = Cx>
struct Concat final : Operator<Scalar>
{
  using Op = Operator<Scalar>;
  using typename Op::Vector;
  using typename Op::Map;
  using typename Op::CMap;

  std::shared_ptr<Op> a, b;

  Concat(std::shared_ptr<Op> a_, std::shared_ptr<Op> b_) :
  a{a_}, b{b_} {
  }

  auto rows() const -> Index { return a->rows(); }
  auto cols() const -> Index { return b->cols(); }

  void forward(CMap const &x, Map &y) const {
    if (a->cols() == b->rows()) {
      a->forward(x, y);
      b->forward(y, y);
    } else {
      Vector temp(a->cols());
      a->forward(x, temp);
      b->forward(temp, y);
    }
  }

  void adjoint(CMap const &y, Map &x) const {
    if (b->cols() == a->rows()) {
      b->adjoint(y, x);
      a->adjoint(x, x);
    } else {
      Vector temp(b->cols());
      a->adjoint(y, temp);
      b->adjoint(temp, x);
    }
  }
};

template <typename Scalar = Cx>
struct VStack final : Operator<Scalar>
{
  using typename Operator<Scalar>::Vector;
  using typename Operator<Scalar>::Map;
  using typename Operator<Scalar>::CMap;

  VStack(std::vector<std::shared_ptr<Operator<Scalar>>> const &o)
    : Operator<Scalar>{"VStack"}
    , ops{o}
  {
    for (auto ii = 1; ii < ops.size(); ii++) {
      if (ops[ii]->cols() != ops[ii - 1]->cols()) {
        Log::Fail("Operators had mismatched number of columns");
      }
    }
  }

  auto rows() const -> Index
  {
    return std::accumulate(ops.begin(), ops.end(), 0, [](auto const &op, int a) { return a + op->rows(); });
  }
  auto cols() const -> Index { return ops.front()->cols(); }

  void forward(CMap const &x, Map &y) const
  {
    Index ir = 0;
    for (auto const &op : ops) {
      op->forward(x, Map(y.data() + ir, op->rows()));
      ir += op->rows();
    }
  }

  void adjoint(CMap const &y, Map &x) const
  {
    Index ir = 0;
    x.setConstant(0.f);
    Vector xa(x.size());
    for (auto const &op : ops) {
      op->adjoint(Map(y.data() + ir, op->rows()), xa);
      x += xa;
      ir += op->rows();
    }
  }

private:
  std::vector<std::shared_ptr<Operator<Scalar>>> ops;
};

}

} // namespace rl
