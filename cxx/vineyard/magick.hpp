#pragma once

#include "colors.hpp"

#include "Magick++.h"

namespace rl {

auto ToMagick(rl::RGBImage const &img, float const rotate) -> Magick::Image;
void ToKitty(Magick::Image &img, bool const scale);

}