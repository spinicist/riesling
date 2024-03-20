#include "types.hpp"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_fft(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string>  oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag                    adj(parser, "R", "Adjoint transform", {"adj", 'a'});
  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>(HD5::Keys::Image);
  auto const  fft = FFT::Make<4, 3>(FirstN<4>(images.dimensions()));
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Cx4 img = images.chip<4>(iv);
    if (adj) {
      fft->reverse(img);
    } else {
      fft->forward(img);
    }
    images.chip<4>(iv) = img;
  }
  auto const  fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Image, images.dimensions(), images.data());
  return EXIT_SUCCESS;
}
