#pragma once

#include "types.hpp"

namespace rl {

namespace Parameters {
auto T1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1T2B1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1B1η(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
}

} // namespace rl
