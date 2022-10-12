#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconRSSOp final : Operator<4, 4>
{
  ReconRSSOp(
    Trajectory const &traj,
    std::string const &ktype,
    float const osamp,
    Index const nC,
    Sz3 const matrix,
    Operator<3, 3> *sdc,
    std::optional<Re2> basis);

  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &y) const -> Input const &;

private:
  std::unique_ptr<Operator<5, 4>> nufft_;
  mutable Input x_;
};
} // namespace rl
