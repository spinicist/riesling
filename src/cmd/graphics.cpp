#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"
#include <Magick++.h>
#include <range/v3/range.hpp>
#include <range/v3/view.hpp>
#include <scn/scn.h>

struct IndexPairReader
{
  void operator()(std::string const &name, std::string const &value, rl::IndexPair &p)
  {
    Index x, y;
    auto  result = scn::scan(value, "{},{}", x, y);
    if (!result) { rl::Log::Fail("Could not read Index Pair for {} from value {} because {}", name, value, result.error()); }
    p.dim = x;
    p.index = y;
  }
};

int main_graphics(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::Positional<std::string> oname(parser, "FILE", "Image file to save");

  args::ValueFlag<std::string> dset(parser, "D", "Dataset (image)", {"dset", 'd'}, "image");
  args::ValueFlag<Index>       nSlices(parser, "N", "Number of slices", {"slices", 'n'}, -1);
  args::ValueFlag<Index>       cols(parser, "C", "Output columns", {"cols"}, 8);

  args::ValueFlag<float> win(parser, "W", "Window scale", {"win"}, 0.9);
  args::Flag grey(parser, "G", "Greyscale", {"grey", 'g'});

  args::ValueFlagList<rl::IndexPair, std::vector, IndexPairReader> chips(parser, "C", "Chip a dimension", {"chip", 'c'});

  args::ValueFlag<int> px(parser, "T", "Thumbnail size in pixels", {"pix", 'p'}, 256);

  // args::ValueFlag<rl::Index> dimS(parser, "S", "Slice dimension (3)", {"slice", 's'}, 3);
  // args::ValueFlag<rl::Index> dimX(parser, "X", "Image X dimension (1)", {"X", 'x'}, 1);
  // args::ValueFlag<rl::Index> dimY(parser, "Y", "Image Y dimension (1)", {"Y", 'y'}, 1);
  ParseCommand(parser, iname);

  Magick::InitializeMagick(NULL);

  rl::HD5::Reader reader(iname.Get());

  auto const diskOrder = reader.order(dset.Get());
  if (diskOrder - chips.Get().size() != 3) {
    rl::Log::Fail("Dataset {} has order {} and only {} chips", dset.Get(), diskOrder, chips.Get().size());
  }

  rl::Cx3     data = reader.readSlab<rl::Cx3>(dset.Get(), chips.Get());
  auto const  dShape = data.dimensions();
  float const maxMag = rl::Maximum(data.abs());
  rl::Log::Print("Data shape {}, max magnitude {}", dShape, maxMag);

  auto const N = nSlices ? std::min(nSlices.Get(), dShape[2]) : dShape[2];
  auto const rows = (Index)std::ceil(N / (float)cols.Get());

  std::vector<Magick::Image> slices, frames;
  for (Index iK = 0; iK < dShape[2]; iK += dShape[2] / N) {
    rl::Cx2    temp = data.chip<2>(iK);
    auto const slice = rl::Colorize(temp, win.Get() * maxMag, grey);
    slices.push_back(Magick::Image(dShape[0], dShape[1], "RGB", Magick::CharPixel, slice.data()));
  }
  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(cols.Get(), rows));
  montageOpts.geometry(Magick::Geometry(px.Get(), px.Get()));
  Magick::montageImages(&frames, slices.begin(), slices.end(), montageOpts);
  if (oname) {
    frames.front().write(oname.Get());
  } else {
    Magick::Blob blob;
    frames.front().magick("PNG");
    frames.front().write(&blob);
    auto const     b64 = blob.base64();
    constexpr auto ChunkSize = 4096;
    if (b64.size() <= ChunkSize) {
      fmt::print(stderr, "\x1B_Ga=T,f=100;{}\x1B\\", b64);
    } else {
      auto const chunks = b64 | ranges::views::chunk(ChunkSize);
      auto const nChunks = chunks.size();
      fmt::print(stderr, "\x1B_Ga=T,f=100,m=1;{}\x1B\\", std::string_view(chunks[0].data(), chunks[0].size()));
      for (int i = 1; i < nChunks - 1; i++) {
        fmt::print(stderr, "\x1B_Gm=1;{}\x1B\\", std::string_view(chunks[i].data(), chunks[i].size()));
      }
      fmt::print(stderr, "\x1B_Gm=0;{}\x1B\\", std::string_view(chunks[nChunks - 1].data(), chunks[nChunks - 1].size()));
    }
    fmt::print(stderr, "\n");
  }
  return EXIT_SUCCESS;
}
