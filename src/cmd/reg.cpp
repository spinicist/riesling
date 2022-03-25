#include "types.h"

#include "algo/llr.h"
#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"

int main_reg(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag llr(parser, "", "Apply sliding-window Locally Low-Rank reg", {"llr"});
  args::Flag llrPatch(parser, "", "Apply patch-based Locally Low-Rank reg", {"llr-patch"});
  args::ValueFlag<float> lambda(
    parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);
  ParseCommand(parser);

  if (!llr && !llrPatch) {
    throw args::Error("Must specify at least one regularization method");
  }

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 const images = input.readTensor<Cx5>(HD5::Keys::Image);
  Cx5 output(images.dimensions());

  for (Index iv = 0; iv < images.dimension(4); iv++) {
    if (llr) {
      output.chip<4>(iv) = llr_sliding(images.chip<4>(iv), lambda.Get(), patchSize.Get());
    } else if (llrPatch.Get()) {
      output.chip<4>(iv) = llr_patch(images.chip<4>(iv), lambda.Get(), patchSize.Get());
    }
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "reg", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(output, HD5::Keys::Image);

  return EXIT_SUCCESS;
}
