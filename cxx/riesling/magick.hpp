#pragma once

#include "rl/colors.hpp"

#include "Magick++.h"

namespace rl {

auto ToMagick(rl::RGBImage const &img, float const rotate) -> Magick::Image;
int  ScreenWidthInPixels();
void ToKitty(Magick::Image &img, bool const scale);
template <int N>
void LiveDebug(std::string const &name, Sz<N> const &shape, Cx const *data, std::array<std::string, N> const &dims);

} // namespace rl