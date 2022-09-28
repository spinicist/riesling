#include "types.hpp"

#include "func/dict.hpp"
#include "func/llr.hpp"
#include "func/slr.hpp"
#include "func/thresh-wavelets.hpp"
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
  args::Flag llr(parser, "", "Apply sliding-window Locally Low-Rank reg", {"llr"});
  args::Flag llrPatch(parser, "", "Apply patch-based Locally Low-Rank reg", {"llr-patch"});
  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);
  args::ValueFlag<std::string> brute(parser, "D", "Brute-force dictionary projection", {"brute"});
  args::ValueFlag<std::string> ball(parser, "D", "Ball-tree dictionary projection", {"ball"});
  args::ValueFlag<float> wavelets(parser, "W", "Wavelet denoising threshold", {"wavelets"});
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);
  args::ValueFlag<Index> levels(parser, "L", "Wavelet levels", {"levels", 'l'}, 4);
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
  Cx5 output(images.dimensions());

  if (wavelets) {
    Sz4 dims = FirstN<4>(images.dimensions());
    ThresholdWavelets tw(dims, width.Get(), levels.Get(), wavelets.Get());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = tw(images.chip<4>(iv));
    }
  } else if (brute) {
    HD5::Reader dictReader(brute.Get());
    BruteForceDictionary dict{dictReader.readMatrix<Eigen::MatrixXf>(HD5::Keys::Dictionary)};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = dict(images.chip<4>(iv));
    }
  } else if (ball) {
    HD5::Reader dictReader(ball.Get());
    BallTreeDictionary dict{dictReader.readMatrix<Eigen::MatrixXf>(HD5::Keys::Dictionary)};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = dict(images.chip<4>(iv));
    }
  } else if (llr || llrPatch) {
    LLR reg{λ.Get(), patchSize.Get(), !llrPatch};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = reg(images.chip<4>(iv));
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
