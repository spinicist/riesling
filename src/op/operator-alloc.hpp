#pragma once

#include "operator.hpp"

/* Linear Operator
 *
 * This is my attempt at some kind of bastardized linear operator struct.
 * The key weirdness here is that the operators track the rank of their inputs/outputs. Technically
 * a linear operator should only be applied to vectors and matrices, but within this context all of
 * those vectors represent higher-rank tensors that are for the purposes of the operator treated as
 * a vector.
 *
 * Hence here we track the input/output ranks.
 */

namespace rl {

template <typename Scalar_, size_t InRank, size_t OutRank>
struct OperatorAlloc : Operator<Scalar_, InRank, OutRank>
{
  OP_INHERIT(Scalar_, InRank, OutRank)

  OperatorAlloc(std::string const &n, InputDims const xd, OutputDims const yd)
    : Parent(n, xd, yd)
    , xStorage_{xd}
    , yStorage_{yd}
    , x_{xStorage_}
    , y_{yStorage_}
  {
    y_.setZero();
    Log::Print<Log::Level::Debug>(FMT_STRING("{} allocated {:L} bytes"), this->name(), (x_.size() + y_.size()) * sizeof(Scalar));
  }

  OperatorAlloc(std::string const &n, InputMap const xm, OutputDims const yd)
    : Parent(n, xm.dimensions(), yd.dimensions())
    , yStorage_{yd}
    , x_{xm}
    , y_{yStorage_}
  {
    y_.setZero();
    Log::Print<Log::Level::Debug>(FMT_STRING("{} allocated {:L} bytes"), this->name(), y_.size() * sizeof(Scalar));
  }

  OperatorAlloc(std::string const &n, InputDims const xd, OutputMap const ym)
    : Parent(n, xd, ym.dimensions())
    , xStorage_{xd}
    , x_{xStorage_}
    , y_{ym}
  {
    x_.setZero();
    Log::Print<Log::Level::Debug>(FMT_STRING("{} allocated {:L} bytes"), this->name(), x_.size() * sizeof(Scalar));
  }

  OperatorAlloc(std::string const &n, InputMap const xm, OutputMap const ym)
    : Parent(n, xm.dimensions(), ym.dimensions())
    , x_{xm}
    , y_{ym}
  {

    Log::Print<Log::Level::Debug>(FMT_STRING("{} allocated 0 bytes"), this->name(), inputDimensions(), outputDimensions());
  }

  virtual ~OperatorAlloc(){};

  auto input() const { return x_; }
  auto output() const { return y_; }

  using Parent::adjoint;
  using Parent::forward;
  using Parent::inputDimensions;
  using Parent::name;
  using Parent::outputDimensions;

private:
  Input xStorage_;
  Output yStorage_;
  InputMap x_;
  OutputMap y_;
};

#define OPALLOC_INHERIT(SCALAR, INRANK, OUTRANK)                                                                               \
  using Parent = OperatorAlloc<SCALAR, INRANK, OUTRANK>;                                                                       \
  using Scalar = typename Parent::Scalar;                                                                                      \
  static const size_t InputRank = Parent::InputRank;                                                                           \
  using Input = typename Parent::Input;                                                                                        \
  using InputMap = typename Parent::InputMap;                                                                                  \
  using InputDims = typename Parent::InputDims;                                                                                \
  static const size_t OutputRank = Parent::OutputRank;                                                                         \
  using Output = typename Parent::Output;                                                                                      \
  using OutputMap = typename Parent::OutputMap;                                                                                \
  using OutputDims = typename Parent::OutputDims;

#define OPALLOC_DECLARE()                                                                                                      \
  using Parent::name;                                                                                                          \
  using Parent::inputDimensions;                                                                                               \
  using Parent::outputDimensions;                                                                                              \
  auto forward(InputMap x) const->OutputMap;                                                                                   \
  auto adjoint(OutputMap x) const->InputMap;                                                                                   \
  using Parent::forward;                                                                                                       \
  using Parent::adjoint;

} // namespace rl
