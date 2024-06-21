#pragma once

#include "types.hpp"

namespace rl {

namespace Parameters {

void CheckSizes(size_t const             N,
                std::vector<float> const defLo,
                std::vector<float> const defHi,
                std::vector<float>      &lo,
                std::vector<float>      &hi);

auto T1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1T2PD(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1T2η(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1B1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1β(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
auto T1β1β2(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf;
} // namespace Parameters

} // namespace rl
