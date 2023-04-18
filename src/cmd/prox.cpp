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

int main_reg(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::Flag llr(parser, "", "Apply sliding-window Locally Low-Rank reg", {"llr"});
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> winSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);
  args::Flag wavelets(parser, "W", "Wavelets", {"wavelets", 'w'});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wave-levels"}, 4);
  args::ValueFlag<Index> waveSize(parser, "W", "Wavelet size (4/6/8)", {"wave-size"}, 6);
  args::ValueFlag<float> l1(parser, "L1", "L1", {"l1"}, 1.f);
  args::ValueFlag<float> maxent(parser, "E", "Entropy", {"maxent"}, 1.f);
  args::ValueFlag<float> nmrent(parser, "E", "Entropy", {"nmrent"}, 1.f);

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
  Cx5 output(images.dimensions());

  using Map = Prox<Cx>::Map;
  using CMap = Prox<Cx>::CMap;

  Sz4 const dims = FirstN<4>(images.dimensions());
  Index const nvox = Product(dims);
  std::shared_ptr<Prox<Cx>> prox;
  if (wavelets) {
    prox = std::make_shared<ThresholdWavelets>(λ.Get(), dims, waveSize.Get(), waveLevels.Get());
  } else if (llr) {
    prox = std::make_shared<LLR>(λ.Get(), patchSize.Get(), winSize.Get(), dims);
  } else if (l1) {
    prox = std::make_shared<SoftThreshold>(λ.Get());
  } else if (maxent) {
    prox = std::make_shared<Entropy>(λ.Get());
  } else if (nmrent) {
    prox = std::make_shared<NMREntropy>(λ.Get());
  } else {
    throw args::Error("Must specify at least one regularization method");
  }
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    CMap im(&images(0, 0, 0, 0, iv), nvox);
    Map om(&output(0, 0, 0, 0, iv), nvox);
    prox->apply(1.f, im, om);
  }

  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, output.dimensions(), output.data());
  return EXIT_SUCCESS;
}
