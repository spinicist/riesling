#pragma once

#include "types.hpp"

namespace rl {

using RGBImage = Eigen::Tensor<unsigned char, 3>;

auto ColorizeComplex(Cx2 const &img, float const max, float const ɣ) -> RGBImage;
auto ColorizeReal(Re2 const &img, float const max, float const ɣ) -> RGBImage;
auto Greyscale(Re2 const &img, float const min, float const max, float const ɣ) -> RGBImage;

} // namespace rl
