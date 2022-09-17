#include "types.hpp"

#include "func/dict.hpp"
#include "func/llr.hpp"
#include "func/slr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_reg(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag llr(parser, "", "Apply sliding-window Locally Low-Rank reg", {"llr"});
  args::Flag llrPatch(parser, "", "Apply patch-based Locally Low-Rank reg", {"llr-patch"});
  args::Flag slr(parser, "", "Apply Structured Low Rank to channel images", {"slr"});
  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);
  args::ValueFlag<std::string> dictPath(parser, "D", "Apply dictionary projection", {"dict"});
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  if (dictPath) {
    HD5::Reader dictReader(dictPath.Get());
    TreeProjection dict{dictReader.readTensor<Re2>(HD5::Keys::Dictionary)};
    Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
    Cx5 output(images.dimensions());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = dict(images.chip<4>(iv));
    }
    writer.writeTensor(output, HD5::Keys::Image);
  } else if (llr || llrPatch) {
    Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
    Cx5 output(images.dimensions());
    LLR reg{λ.Get(), patchSize.Get(), !llrPatch};
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      output.chip<4>(iv) = reg(images.chip<4>(iv));
    }
    writer.writeTensor(output, HD5::Keys::Image);
  } else if (slr) {
    Cx6 const channels = input.readTensor<Cx6>(HD5::Keys::Channels);
    Cx6 output(channels.dimensions());
    FFTOp<5> fft(FirstN<5>(channels.dimensions()));
    SLR reg{fft, patchSize.Get(), λ.Get()};
    for (Index iv = 0; iv < channels.dimension(5); iv++) {
      output.chip<5>(iv) = reg(channels.chip<5>(iv));
    }
    writer.writeTensor(output, HD5::Keys::Channels);
  } else {
    throw args::Error("Must specify at least one regularization method");
  }

  return EXIT_SUCCESS;
}
