#include "types.hpp"

#include "func/dict.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "prox/entropy.hpp"
#include "prox/llr.hpp"
#include "prox/thresh-wavelets.hpp"
#include "threads.hpp"

using namespace rl;

int main_prox(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> scale(parser, "S", "Scale data before applying prox", {'s', "scale"}, 1);
  args::ValueFlag<float> l1(parser, "L1", "Simple L1 regularization", {"l1"});
  args::ValueFlag<float> nmrent(parser, "E", "NMR Entropy", {"nmrent"});

  args::ValueFlag<float> llr(parser, "L", "LLR regularization", {"llr"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);

  args::ValueFlag<Index> wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wavelet-levels"}, 4);
  args::ValueFlag<Index> waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6);

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image) * Cx(scale.Get());
  Cx5 output(images.dimensions());

  using Map = Prox<Cx>::Map;
  using CMap = Prox<Cx>::CMap;

  Sz4 const dims = FirstN<4>(images.dimensions());
  Index const nvox = Product(dims);
  std::shared_ptr<Prox<Cx>> prox;
  if (wavelets) {
    prox = std::make_shared<ThresholdWavelets>(wavelets.Get(), dims, waveWidth.Get(), waveLevels.Get());
  } else if (llr) {
    prox = std::make_shared<LLR>(llr.Get(), llrPatch.Get(), llrWin.Get(), dims);
  } else if (l1) {
    prox = std::make_shared<SoftThreshold>(l1.Get());
  } else if (nmrent) {
    prox = std::make_shared<Entropy>(nmrent.Get());
  } else {
    throw args::Error("Must specify at least one regularization method");
  }
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    CMap im(&images(0, 0, 0, 0, iv), nvox);
    Map om(&output(0, 0, 0, 0, iv), nvox);
    prox->apply(1.f, im, om);
    om = om / scale.Get();
  }

  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, output.dimensions(), output.data());
  return EXIT_SUCCESS;
}
