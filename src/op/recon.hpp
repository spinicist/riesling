#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconOp final : Operator<4, 4>
{
  ReconOp(
    Trajectory const &traj,
    std::string const &ktype,
    float const osamp,
    Cx4 const &maps,
    Functor<Cx3> *sdc,
    std::optional<Re2> basis,
    bool toeplitz = false);

  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &y) const -> Input const &;
  auto adjfwd(Input const &x) const -> Input;

private:
  std::unique_ptr<Operator<5, 4>> nufft_;
  SenseOp sense_;
};
} // namespace rl
