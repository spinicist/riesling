#pragma once

#include <tuple>

namespace rl {

auto StableGivens(float const a, float const b) -> std::tuple<float, float, float>;

} // namespace rl
