#pragma once

#include "operator.hpp"
#include "trajectory.hpp"

#include <memory>
#include <optional>

namespace rl {

// So we can template the kernel size and still stash pointers
template <typename Scalar, size_t NDim>
struct GridBase : Operator<NDim + 2, 3, Scalar>
{
  using typename Operator<NDim + 2, 3, Scalar>::Input;
  using typename Operator<NDim + 2, 3, Scalar>::Output;

  GridBase()
  {
  }

  virtual ~GridBase(){};
  virtual auto apodization(Sz<NDim> const sz) const -> Eigen::Tensor<float, NDim> = 0;
  virtual Sz3 outputDimensions() const = 0;
  virtual typename Input::Dimensions inputDimensions() const = 0;
  virtual std::shared_ptr<Input> workspace() const = 0;
};

} // namespace rl
