#pragma once
#include "types.hpp"

namespace rl {

using RGBImage = Eigen::Tensor<unsigned char, 3>;
using RGBAImage = Eigen::Tensor<uint32_t, 2>;

auto ToRGB(Re2 const &slice) -> RGBImage;
auto ToRGBA(Re2 const &slice) -> RGBAImage;
auto EncodeRGB(RGBImage const &slice, bool const scale) -> std::vector<std::string>;

}
