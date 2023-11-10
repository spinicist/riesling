#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_wavelets(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::ValueFlag<Sz4, SzReader<4>> dims(parser, "W", "Wavelet dimensions", {"dims"}, Sz4{0, 1, 1, 1});
  args::ValueFlag<Index>            width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  args::Flag adj(parser, "A", "Apply adjoint operation", {"adj"});
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader reader(iname.Get());
  auto const  fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  auto        images = reader.readTensor<Cx5>(HD5::Keys::Image);
  Wavelets    wav(FirstN<4>(images.dimensions()), width.Get(), dims.Get());
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    if (adj) {
      images.chip<4>(iv) = wav.adjoint(ChipMap(images, iv));
    } else {
      images.chip<4>(iv) = wav.forward(ChipMap(images, iv));
    }
  }
  HD5::Writer writer(fname);
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(HD5::Keys::Image, images.dimensions(), images.data());

  return EXIT_SUCCESS;
}
