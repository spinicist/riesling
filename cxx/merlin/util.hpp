#pragma once

#include "rl/info.hpp"
#include "rl/types.hpp"
#include "types.hpp"

namespace merlin {
auto Import(rl::Re3Map const data, rl::Info const info) -> ImageType::Pointer;
auto ITKToRIESLING(TransformType::Pointer t) -> rl::Transform;
} // namespace merlin
