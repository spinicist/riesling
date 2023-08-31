#include "rgb.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

auto ToRGB(Re2 const &slice) -> RGBImage
{
  auto const nX = slice.dimension(0);
  auto const nY = slice.dimension(1);
  auto       rgb = RGBImage(3, nX, nY);
  for (int iy = 0; iy < nY; iy++) {
    for (int ix = 0; ix < nX; ix++) {
      auto const val = static_cast<char>(slice(ix, iy) * 255.f);
      rgb(0, ix, iy) = val;
      rgb(1, ix, iy) = val;
      rgb(2, ix, iy) = val;
    }
  }
  return rgb;
}

auto ToRGBA(Re2 const &slice) -> RGBAImage
{
  auto const nX = slice.dimension(0);
  auto const nY = slice.dimension(1);
  auto       rgba = RGBAImage(nX, nY);
  for (int iy = 0; iy < nY; iy++) {
    for (int ix = 0; ix < nX; ix++) {
      auto const val = static_cast<uint32_t>(slice(ix, iy) * 255.f);
      rgba(ix, iy) = (val) | (val << 8L) | (val << 16L) | (255 << 24L);
    }
  }
  return rgba;
}

} // namespace rl
