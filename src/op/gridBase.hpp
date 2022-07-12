#pragma once

#include "kernel.h"
#include "mapping.h"
#include "operator.hpp"

#include <memory>

namespace rl {

// So we can template the kernel size and still stash pointers
template <typename Scalar>
struct GridBase : Operator<5, 3, Scalar>
{
  using typename Operator<5, 3, Scalar>::Input;
  using typename Operator<5, 3, Scalar>::Output;

  GridBase(Mapping const &map, Index const nC, Index const d1)
    : mapping_{map}
    , inputDims_{AddFront(map.cartDims, nC, d1)}
    , outputDims_{AddFront(map.noncartDims, nC)}
    , ws_{std::make_shared<Input>(inputDims_)}
    , weightFrames_{true}
  {
  }

  virtual ~GridBase(){};
  virtual Output A(Input const &cart) const = 0;
  virtual Input &Adj(Output const &noncart) const = 0;
  virtual R3 apodization(Sz3 const sz) const = 0;

  Sz3 outputDimensions() const override
  {
    return outputDims_;
  }

  Sz5 inputDimensions() const override
  {
    return inputDims_;
  }

  std::shared_ptr<Input> workspace() const
  {
    return ws_;
  }

  void doNotWeightFrames()
  {
    weightFrames_ = false;
  }

  Mapping const &mapping() const
  {
    return mapping_;
  }

protected:
  Mapping mapping_;
  Sz5 inputDims_;
  Sz3 outputDims_;
  std::shared_ptr<Input> ws_;
  bool safe_, weightFrames_;
};

template <typename Scalar>
std::unique_ptr<GridBase<Scalar>>
make_grid(Kernel const *k, Mapping const &m, Index const nC, std::string const &basis = "");

} // namespace rl
