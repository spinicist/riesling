#pragma once

#include "types.hpp"

namespace rl {

using RGBImage = Eigen::Tensor<unsigned char, 3>;

auto Colorize(Cx2 const &img, float const max, bool const grey, float const ɣ) -> RGBImage;

}
