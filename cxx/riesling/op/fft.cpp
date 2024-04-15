#include "types.hpp"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

void main_fft(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::Flag                    adj(parser, "R", "Adjoint transform", {"adj", 'a'});
  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>();
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
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data());
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
