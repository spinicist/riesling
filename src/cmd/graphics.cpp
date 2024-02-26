#include "basis/basis.hpp"
#include "colors.hpp"
#include "io/hd5.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "types.hpp"
#include <Magick++.h>

int main_graphics(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to slice");
  args::Positional<std::string> oname(parser, "FILE", "Image file to save");

  args::ValueFlag<std::string> dset(parser, "D", "Dataset (image)", {"dset", 'd'}, "image");
  args::ValueFlag<Index> nSlices(parser, "N", "Number of slices", {"slices", 'n'}, -1);
  args::ValueFlag<Index> cols(parser, "C", "Output columns", {"cols"}, 8);

  args::ValueFlag<float> win(parser, "W", "Window scale", {"win"}, 0.9);

  // args::ValueFlag<rl::Index> dimS(parser, "S", "Slice dimension (3)", {"slice", 's'}, 3);
  // args::ValueFlag<rl::Index> dimX(parser, "X", "Image X dimension (1)", {"X", 'x'}, 1);
  // args::ValueFlag<rl::Index> dimY(parser, "Y", "Image Y dimension (1)", {"Y", 'y'}, 1);
  ParseCommand(parser);

  Magick::InitializeMagick(NULL);

  rl::HD5::Reader reader(iname.Get());
  rl::Cx3 data = reader.readSlab<rl::Cx3>(dset.Get(), {0, 4}, {0, 0});
  auto const dShape = data.dimensions();
  float const maxMag = rl::Maximum(data.abs());
  rl::Log::Print("Data shape {}, max magnitude {}", dShape, maxMag);

  auto const N = nSlices ? std::min(nSlices.Get(), dShape[2]) : dShape[2];
  auto const rows = (Index)std::ceil(N / (float)cols.Get());

  Magick::Montage montageOpts;
  montageOpts.backgroundColor(Magick::Color(0, 0, 0));
  montageOpts.tile(Magick::Geometry(cols.Get(), rows));

  std::vector<Magick::Image> slices, frames;
  for (Index iK = 0; iK < dShape[2]; iK += dShape[2]/N) {
    rl::Cx2 temp = data.chip<2>(iK);
    auto const slice = rl::Colorize(temp, win.Get() * maxMag, false);
    slices.push_back(Magick::Image(dShape[0], dShape[1], "RGB", Magick::CharPixel, slice.data()));
  }
  Magick::montageImages(&frames, slices.begin(), slices.end(), montageOpts);
  frames.front().write(oname.Get());
  rl::Log::Print("Wrote output file {}", oname.Get());
  return EXIT_SUCCESS;
}
