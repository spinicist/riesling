#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_transform(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::ValueFlag<Index> wavelets(parser, "W", "Wavelet denoising levels", {"wavelets"}, 4);
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  args::Flag grad(parser, "G", "Grad / Div", {"grad"});

  args::Flag fwd(parser, "F", "Apply forward operation", {"fwd"});
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }

  HD5::Reader reader(iname.Get());
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(reader.readInfo());

  if (wavelets) {

    auto images = reader.readTensor<Cx5>(HD5::Keys::Image);
    Wavelets wav(FirstN<4>(images.dimensions()), width.Get(), wavelets.Get());
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      if (fwd) {
        wav.forward(ChipMap(images, iv));
      } else {
        wav.adjoint(ChipMap(images, iv));
      }
    }
    writer.writeTensor(images, HD5::Keys::Image);
  } else if (grad) {
    if (fwd) {
      auto input = reader.readTensor<Cx5>(HD5::Keys::Image);
      Sz4 dims = FirstN<4>(input.dimensions());
      Cx6 output(AddBack(dims, 3, input.dimension(4)));
      GradOp g(dims);
      for (Index iv = 0; iv < input.dimension(4); iv++) {
        output.chip<5>(iv) = g.cforward(CChipMap(input, iv));
      }
      writer.writeTensor(output, "grad");
    } else {
      auto input = reader.readTensor<Cx6>("grad");
      Sz4 dims = FirstN<4>(input.dimensions());
      Cx5 output(AddBack(dims, input.dimension(5)));
      GradOp g(dims);
      for (Index iv = 0; iv < input.dimension(5); iv++) {
        output.chip<4>(iv) = g.cadjoint(CChipMap(input, iv));
      }
      writer.writeTensor(output, HD5::Keys::Image);
    }
  } else {
    Log::Fail("A transform option must be specified");
  }

  return EXIT_SUCCESS;
}
