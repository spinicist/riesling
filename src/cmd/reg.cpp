#include "types.hpp"

#include "func/dict.hpp"
#include "prox/llr.hpp"
#include "prox/slr.hpp"
#include "prox/thresh-wavelets.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
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
  args::ValueFlag<std::string> brute(parser, "D", "Brute-force dictionary projection", {"brute"});
  args::ValueFlag<std::string> ball(parser, "D", "Ball-tree dictionary projection", {"ball"});
  args::Flag wavelets(parser, "W", "Wavelets", {"wavelets", 'w'});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wave-levels"}, 4);
  args::ValueFlag<Index> waveSize(parser, "W", "Wavelet size (4/6/8)", {"wave-size"}, 6);
  args::ValueFlag<float> l1(parser, "L1", "L1", {"l1"}, 1.f);

  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
  Cx5 output(images.dimensions());

  if (wavelets) {
    Sz4 dims = FirstN<4>(images.dimensions());
    ThresholdWavelets tw(dims, λ.Get(), waveSize.Get(), waveLevels.Get());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = tw(1.f, CChipMap(images, iv));
    }
  } else if (brute) {
    HD5::Reader dictReader(brute.Get());
    BruteForceDictionary dict{dictReader.readMatrix<Eigen::MatrixXf>(HD5::Keys::Dictionary)};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
       dict(CChipMap(images, iv), ChipMap(output, iv));
    }
  } else if (ball) {
    HD5::Reader dictReader(ball.Get());
    BallTreeDictionary dict{dictReader.readMatrix<Eigen::MatrixXf>(HD5::Keys::Dictionary)};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      dict(CChipMap(images, iv), ChipMap(output, iv));
    }
  } else if (llr) {
    LLR reg(λ.Get(), patchSize.Get(), winSize.Get());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = reg(λ.Get(), CChipMap(images, iv));
    }
  } else if (l1) {
    SoftThreshold<Cx4> thresh(λ.Get());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = thresh(l1.Get(), CChipMap(images, iv));
    }
  } else {
    throw args::Error("Must specify at least one regularization method");
  }
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(output, HD5::Keys::Image);
  return EXIT_SUCCESS;
}
