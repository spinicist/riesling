#pragma once

#include "kernel.hpp"
#include "mapping.h"
#include "operator.hpp"

#include <memory>
#include <optional>

namespace rl {

// So we can template the kernel size and still stash pointers
template <typename Scalar>
struct GridBase : Operator<5, 3, Scalar>
{
  using typename Operator<5, 3, Scalar>::Input;
  using typename Operator<5, 3, Scalar>::Output;

  GridBase()
    : weightFrames_{true}
  {
  }

  virtual ~GridBase(){};
  virtual Output forward(Input const &cart) const = 0;
  virtual Input &adjoint(Output const &noncart) const = 0;
  virtual Re3 apodization(Sz3 const sz) const = 0;
  virtual Sz3 outputDimensions() const = 0;
  virtual Sz5 inputDimensions() const = 0;
  virtual std::shared_ptr<Input> workspace() const = 0;

  void doNotWeightFrames()
  {
    weightFrames_ = false;
  }

protected:
  bool weightFrames_;
};

template <typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid(
  Trajectory const &trajectory,
  std::string const kType,
  float const os,
  Index const nC,
  std::optional<Re2> const &basis = std::nullopt);

} // namespace rl
