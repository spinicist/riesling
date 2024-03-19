#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

#include <Magick++.h>
#include <ranges>
#include <scn/scn.h>
#include <sys/ioctl.h>
#include <tl/chunk.hpp>
#include <tl/to.hpp>

struct IndexPairReader
{
  void operator()(std::string const &name, std::string const &value, IndexPair &p)
  {
    Index x, y;
    auto  result = scn::scan(value, "{},{}", x, y);
    if (!result) { rl::Log::Fail("Could not read Index Pair for {} from value {}", name, value); }
    p.dim = x;
    p.index = y;
  }
};

auto ReadData(std::string const &iname, std::string const &dset, std::vector<IndexPair> chips) -> rl::Cx3
{
  rl::HD5::Reader reader(iname);
  auto const      diskOrder = reader.order(dset);
  auto const      diskDims = reader.dimensions(dset);
  if (!chips.size()) {
    if (dset == "image") {
      chips = std::vector<IndexPair>{{0, 0}, {4, 0}};
    } else if (dset == "noncartesian") {
      chips = std::vector<IndexPair>{{3, 0}, {4, 0}};
    } else if (dset == "sense") {
      chips = std::vector<IndexPair>{{1, 0}, {4, diskDims.at(4) / 2}};
    }
  } else {
    if (diskOrder - chips.size() != 3) {
      rl::Log::Fail("Dataset {} has order {} and only {} chips", dset, diskOrder, chips.size());
    }
  }
  rl::Log::Debug("Reading chips: {}", chips);
  return reader.readSlab<rl::Cx3>(dset, chips);
}

auto SliceData(rl::Cx3 const &data,
               Index const    slDim,
               Index const    slStart,
               Index const    slEnd,
               Index const    slN,
               float const    win,
               bool const     grey,
               bool const     log) -> std::vector<Magick::Image>
{
  auto const dShape = data.dimensions();
  if (slDim < 0 || slDim > 2) { rl::Log::Fail("Slice dim was {}, must be 0-2", slDim); }
  if (slStart < 0 || slStart >= dShape[slDim]) { rl::Log::Fail("Slice start invalid"); }
  if (slEnd && slEnd >= dShape[slDim]) { rl::Log::Fail("Slice end invalid"); }
  if (slN < 0) { rl::Log::Fail("Requested negative number of slices"); }
  auto const start = slStart;
  auto const end = slEnd ? slEnd : dShape[slDim];
  auto const maxN = end - start;
  auto const N = slN ? std::min(slN, maxN) : maxN;
  float const stride = maxN / (float)(N - 1.f);
  std::vector<Magick::Image> slices;
  for (float iK = start; iK < end; iK += stride) {
    rl::Cx2    temp = data.chip(std::floor(iK), slDim);
    auto const slice = rl::Colorize(temp, win, grey, log);
    slices.push_back(Magick::Image(dShape[(slDim + 1) % 3], dShape[(slDim + 2) % 3], "RGB", Magick::CharPixel, slice.data()));
    slices.back().flip(); // Reverse Y for display
  }
  return slices;
}

auto DoMontage(std::vector<Magick::Image> &slices, Index const cols, Index const px) -> Magick::Image
{
  auto const      rows = (Index)std::ceil(slices.size() / (float)cols);
  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(cols, rows));
  montageOpts.geometry(slices.front().size());
  std::vector<Magick::Image> frames;
  Magick::montageImages(&frames, slices.begin(), slices.end(), montageOpts);

  return frames.front();
}

void Kittify(Magick::Image &graphic)
{
  struct winsize winSize;
  ioctl(0, TIOCGWINSZ, &winSize);
  auto const scaling = winSize.ws_xpixel / (float)graphic.size().width();
  graphic.resize(Magick::Geometry(winSize.ws_xpixel, scaling * graphic.size().height()));
  Magick::Blob blob;
  graphic.write(&blob);
  auto const     b64 = blob.base64();
  constexpr auto ChunkSize = 4096;
  if (b64.size() <= ChunkSize) {
    fmt::print(stderr, "\x1B_Ga=T,f=100;{}\x1B\\", b64);
  } else {
    auto const chunks = b64 | tl::views::chunk(ChunkSize);
    auto const nChunks = chunks.size();
    fmt::print(stderr, "\x1B_Ga=T,f=100,m=1;{}\x1B\\", std::string_view(chunks.front().data(), chunks.front().size()));
    for (auto &&chunk : chunks | std::ranges::views::drop(1) | std::ranges::views::take(nChunks - 2)) {
      fmt::print(stderr, "\x1B_Gm=1;{}\x1B\\", std::string_view(chunk.data(), chunk.size()));
    }
    fmt::print(stderr, "\x1B_Gm=0;{}\x1B\\", std::string_view(chunks.back().data(), chunks.back().size()));
  }
  fmt::print(stderr, "\n");
}

int main_montage(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::Positional<std::string> oname(parser, "FILE", "Image file to save");
  args::ValueFlag<std::string>  dset(parser, "D", "Dataset (image)", {"dset", 'd'}, "image");

  args::ValueFlagList<IndexPair, std::vector, IndexPairReader> chips(parser, "C", "Chip a dimension", {"chip", 'c'});

  args::ValueFlag<Index> cols(parser, "C", "Output columns", {"cols"}, 8);
  args::ValueFlag<int>   px(parser, "T", "Thumbnail size in pixels", {"pix", 'p'}, 256);

  args::ValueFlag<float> max(parser, "W", "Max intensity", {"max"});
  args::ValueFlag<float> maxP(parser, "P", "Max intensity as %", {"maxP"}, 0.9);
  args::Flag             grey(parser, "G", "Greyscale", {"grey", 'g'});
  args::Flag             log(parser, "L", "Logarithmic intensity", {"log", 'l'});

  args::ValueFlag<Index> slN(parser, "N", "Number of slices (0 for all)", {"num", 'n'}, 8);
  args::ValueFlag<Index> slStart(parser, "S", "Start slice", {"start"}, 0);
  args::ValueFlag<Index> slEnd(parser, "S", "End slice", {"end"});
  args::ValueFlag<Index> slDim(parser, "S", "Slice dimension (0/1/2)", {"dim"}, 0);

  ParseCommand(parser, iname);
  Magick::InitializeMagick(NULL);

  auto const  data = ReadData(iname.Get(), dset.Get(), chips.Get());
  float const maxData = rl::Maximum(data.abs());
  float const winMax = max ? max.Get() : maxP.Get() * maxData;
  rl::Log::Print("Max magnitude in data {}. Window maximum {}", maxData, winMax);

  auto slices = SliceData(data, slDim.Get(), slStart.Get(), slEnd.Get(), slN.Get(), winMax, grey, log);
  auto graphic = DoMontage(slices, cols.Get(), px.Get());
  graphic.magick("PNG");
  if (oname) {
    graphic.write(oname.Get());
  } else {
    Kittify(graphic);
  }
  return EXIT_SUCCESS;
}
