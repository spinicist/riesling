#pragma once

#include "rl/info.hpp"
#include "rl/types.hpp"
#include "types.hpp"

namespace merlin {
auto Register(ImageType::Pointer fixed, ImageType::Pointer moving, ImageType::RegionType mask) -> TransformType::Pointer;
} // namespace merlin
