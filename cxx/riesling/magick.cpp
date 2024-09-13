#include "magick.hpp"

#include "log.hpp"
#include "tensors.hpp"

#include <fmt/format.h>
#include <ranges>
#include <sys/ioctl.h>
#include <tl/chunk.hpp>
#include <tl/to.hpp>

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
  float const iscale = img.size().height() / (float)img.size().width();
  float const cscale = (winSize.ws_xpixel / (float)winSize.ws_col) / (winSize.ws_ypixel / (float)winSize.ws_row);
  Index const rows = winSize.ws_col * iscale * cscale;
  auto const  scHdr = scale ? fmt::format(",c={},r={}", winSize.ws_col, rows) : "";
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
    fmt::print(stderr, "\x1B_Ga=T,f=100,m=1{};{}\x1B\\", scHdr, std::string_view(chunks.front().data(), chunks.front().size()));
    for (auto &&chunk : chunks | std::ranges::views::drop(1) | std::ranges::views::take(nChunks - 2)) {
      fmt::print(stderr, "\x1B_Gm=1;{}\x1B\\", std::string_view(chunk.data(), chunk.size()));
    }
    fmt::print(stderr, "\x1B_Gm=0;{}\x1B\\", std::string_view(chunks.back().data(), chunks.back().size()));
  }
  fmt::print(stderr, "\n");
}

template <int N>
auto LiveSlice(Eigen::TensorMap<CxN<N> const> const &xt,
               std::array<std::string, N> const     &dimNames,
               std::string const                    &d0,
               std::string const                    &d1) -> Magick::Image
{
  auto const shape = xt.dimensions();
  Sz<N>      st, sz = shape;
  Sz2        shape2;
  for (Index id = 0; id < N; id++) {
    if (dimNames[id] == d0) {
      st[id] = 0;
      shape2[0] = shape[id];
    } else if (dimNames[id] == d1) {
      st[id] = 0;
      shape2[1] = shape[id];
    } else {
      st[id] = shape[id] / 2;
      sz[id] = 1;
    }
  }
  if (shape2[0] > 0 && shape2[1] > 0) {
    rl::CxN<N> const            sliced = xt.slice(st, sz);
    Eigen::TensorMap<Cx2 const> slice(sliced.data(), shape2);
    auto const                  win = Maximum(slice.abs());
    auto                        magick = ToMagick(ColorizeComplex(slice, win, 0.8), 0.f);
    return magick;
  } else {
    Log::Fail("magick", "Could not find dimensions {} and {}", d0, d1);
  }
}

template <int N>
void LiveDebug(std::string const &, Sz<N> const &shape, Cx const *data, std::array<std::string, N> const &dimNames)
{
  Magick::InitializeMagick(NULL);
  auto const xt = Eigen::TensorMap<CxN<N> const>(data, shape);

  std::vector<Magick::Image> magicks;
  magicks.push_back(LiveSlice<N>(xt, dimNames, "i", "j"));
  magicks.push_back(LiveSlice<N>(xt, dimNames, "j", "k"));
  magicks.push_back(LiveSlice<N>(xt, dimNames, "k", "i"));

  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(3, 1));
  montageOpts.geometry(magicks.front().size());
  std::vector<Magick::Image> frames;
  Magick::montageImages(&frames, magicks.begin(), magicks.end(), montageOpts);
  ToKitty(frames.front(), true);
}

template void LiveDebug<4>(std::string const &, Sz4 const &, Cx const *, std::array<std::string, 4> const &);

} // namespace rl