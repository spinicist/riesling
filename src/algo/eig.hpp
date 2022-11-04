#pragma once

#include "log.hpp"
#include "tensorOps.hpp"
#include <optional>

namespace rl {

template <typename Op>
auto PowerMethodForward(std::shared_ptr<Op> op, Index const iterLimit, std::optional<Cx4> const &P)
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;
  Log::Print("Power Method for A'A");
  Input vec(op->inputDimensions());
  vec.template setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
  float val = Norm(vec);
  vec /= vec.constant(val);

  for (auto ii = 0; ii < iterLimit; ii++) {
    Output o = op->forward(vec);
    if (P) {
      o *= *P;
    }
    vec = op->adjoint(o);
    val = Norm(vec);
    vec /= vec.constant(val);
    Log::Print<Log::Level::High>(FMT_STRING("Iteration {} Eigenvalue {}"), ii, val);
  }

  return std::make_tuple(val, vec);
}

template <typename Op>
auto PowerMethodAdjoint(std::shared_ptr<Op> op, Index const iterLimit, std::optional<Cx4> const &P)
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;
  Output vec(op->outputDimensions());
  vec.template setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
  float val = Norm(vec);
  vec /= vec.constant(val);
  Log::Print(FMT_STRING("Power Method for adjoint system (AA')"));
  for (auto ii = 0; ii < iterLimit; ii++) {
    Input i = op->adjoint(vec);
    vec = op->forward(i);
    if (P) {
      vec *= *P;
    }
    val = Norm(vec);
    vec /= vec.constant(val);
    Log::Print<Log::Level::High>(FMT_STRING("Iteration {} Eigenvalue {}"), ii, val);
  }

  return std::make_tuple(val, vec);
}

} // namespace rl
