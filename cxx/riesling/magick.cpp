#include "magick.hpp"

#include <ranges>
#include <sys/ioctl.h>
#include <tl/chunk.hpp>
#include <tl/to.hpp>
#include <fmt/format.h>

namespace rl {

auto ToMagick(rl::RGBImage const &img, float const rotate) -> Magick::Image
{
  Magick::Image tmp(img.dimension(1), img.dimension(2), "RGB", Magick::CharPixel, img.data());
  tmp.flip();
  tmp.rotate(rotate);
  return tmp;
}

void ToKitty(Magick::Image &img, bool const scale)
{
  struct winsize winSize;
  ioctl(0, TIOCGWINSZ, &winSize);
  float const  iscale = img.size().height() / (float)img.size().width();
  float const  cscale = (winSize.ws_xpixel / (float)winSize.ws_col) / (winSize.ws_ypixel / (float)winSize.ws_row);
  Index const  rows = winSize.ws_col * iscale * cscale;
  auto const scHdr = scale ? fmt::format(",c={},r={}", winSize.ws_col, rows) : "";
  img.magick("PNG");
  Magick::Blob blob;
  img.write(&blob);
  auto const     b64 = blob.base64();
  constexpr auto ChunkSize = 4096;
  if (b64.size() <= ChunkSize) {
    fmt::print(stderr, "\x1B_Ga=T,f=100{};{}\x1B\\", scHdr, b64);
  } else {
    auto const chunks = b64 | tl::views::chunk(ChunkSize);
    auto const nChunks = chunks.size();
    fmt::print(stderr, "\x1B_Ga=T,f=100,m=1{};{}\x1B\\", scHdr,
               std::string_view(chunks.front().data(), chunks.front().size()));
    for (auto &&chunk : chunks | std::ranges::views::drop(1) | std::ranges::views::take(nChunks - 2)) {
      fmt::print(stderr, "\x1B_Gm=1;{}\x1B\\", std::string_view(chunk.data(), chunk.size()));
    }
    fmt::print(stderr, "\x1B_Gm=0;{}\x1B\\", std::string_view(chunks.back().data(), chunks.back().size()));
  }
  fmt::print(stderr, "\n");
}

} // namespace rl