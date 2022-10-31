#pragma once

#include "tensorOps.hpp"
#include "log.hpp"

namespace rl {

template <typename Op>
auto PowerMethod(std::shared_ptr<Op> op, Index const iterLimit)
{
  using Input = typename Op::Input;
  Log::Print("Estimating largest eigenvalue with power method");
  Input vec(op->inputDimensions());
  vec.template setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
  float val;

  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = op->adjfwd(vec);
    val = Norm(vec);
    vec /= vec.constant(val);
    Log::Print<Log::Level::High>(FMT_STRING("Iteration {} Eigenvalue {}"), ii, val);
  }

  return std::make_tuple(val, vec);
}

}
