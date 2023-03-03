#pragma once

#include "../log.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

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

template <typename Scalar_, size_t InRank, size_t OutRank = InRank>
struct Operator
{
  static const size_t InputRank = InRank;
  using Scalar = Scalar_;
  using Input = Eigen::Tensor<Scalar, InputRank>;
  using InputMap = Eigen::TensorMap<Input>;
  using CInputMap = Eigen::TensorMap<Input const>;
  using InputDims = typename Input::Dimensions;
  static const size_t OutputRank = OutRank;
  using Output = Eigen::Tensor<Scalar, OutputRank>;
  using OutputMap = Eigen::TensorMap<Output>;
  using COutputMap = Eigen::TensorMap<Output const>;
  using OutputDims = typename Output::Dimensions;

  Operator(std::string const &name, InputDims const xd, OutputDims const yd)
    : name_{name}
    , xDims_{xd}
    , yDims_{yd}
  {
    Log::Print<Log::Level::Debug>(
      FMT_STRING("{} created. Input dims {} Output dims {}"),
      name_,
      inputDimensions(),
      outputDimensions());
  }

  virtual ~Operator(){};

  auto name() const { return name_; };
  auto inputDimensions() const { return xDims_; };
  auto outputDimensions() const { return yDims_; };

  virtual auto forward(InputMap x) const -> OutputMap = 0;
  virtual auto adjoint(OutputMap y) const -> InputMap = 0;
  virtual auto adjfwd(InputMap x) const -> InputMap { Log::Fail("AdjFwd Not implemented"); }

  virtual auto cforward(CInputMap x) const -> Output
  {
    Input xcopy = x;
    return this->forward(InputMap(xcopy));
  }

  virtual auto cadjoint(COutputMap y) const -> Input
  {
    Output ycopy = y;
    return this->adjoint(OutputMap(ycopy));
  }

  auto startForward(InputMap x) const
  {
    if (x.dimensions() != inputDimensions()) {
      Log::Fail("{} forward dims were: {} expected: {}", name_, x.dimensions(), inputDimensions());
    }
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Print<Log::Level::Debug>(FMT_STRING("{} forward started. Norm {}"), name_, Norm(x));
    }
    return Log::Now();
  }

  void finishForward(OutputMap y, Log::Time const start) const
  {
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Print<Log::Level::Debug>(FMT_STRING("{} forward finished. Took {}. Norm {}."), name_, Log::ToNow(start), Norm(y));
    }
  }

  auto startAdjoint(OutputMap y) const
  {
    if (y.dimensions() != outputDimensions()) {
      Log::Fail("{} adjoint dims were: {} expected: {}", name_, y.dimensions(), outputDimensions());
    }
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Print<Log::Level::Debug>(FMT_STRING("{} adjoint started. Norm {}"), name_, Norm(y));
    }
    return Log::Now();
  }

  void finishAdjoint(InputMap x, Log::Time const start) const
  {
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Print<Log::Level::Debug>(FMT_STRING("{} adjoint finished. Took {}. Norm {}"), name_, Log::ToNow(start), Norm(x));
    }
  }

private:
  std::string name_;
  InputDims xDims_;
  OutputDims yDims_;
};

#define OP_INHERIT(SCALAR, INRANK, OUTRANK)                                                                                    \
  using Parent = Operator<SCALAR, INRANK, OUTRANK>;                                                                            \
  using Scalar = typename Parent::Scalar;                                                                                      \
  static const size_t InputRank = Parent::InputRank;                                                                           \
  using Input = typename Parent::Input;                                                                                        \
  using InputMap = typename Parent::InputMap;                                                                                  \
  using InputDims = typename Parent::InputDims;                                                                                \
  static const size_t OutputRank = Parent::OutputRank;                                                                         \
  using Output = typename Parent::Output;                                                                                      \
  using OutputMap = typename Parent::OutputMap;                                                                                \
  using OutputDims = typename Parent::OutputDims;

#define OP_DECLARE()                                                                                                           \
  using Parent::name;                                                                                                          \
  using Parent::inputDimensions;                                                                                               \
  using Parent::outputDimensions;                                                                                              \
  auto forward(InputMap x) const->OutputMap;                                                                                   \
  auto adjoint(OutputMap x) const->InputMap;                                                                                   \
  using Parent::forward;                                                                                                       \
  using Parent::adjoint;

} // namespace rl
