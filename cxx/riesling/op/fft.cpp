#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
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
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  auto const order = input.order();
  switch (order) {
  case 5: {
    Cx5        images = input.readTensor<Cx5>();
    auto const fft = TOps::FFT<4, 3>(FirstN<4>(images.dimensions()));
    for (Index iv = 0; iv < images.dimension(4); iv++) {
      Cx4 img = images.chip<4>(iv);
      if (adj) {
        img = fft.adjoint(img);
      } else {
        img = fft.forward(img);
      }
      images.chip<4>(iv) = img;
    }
    writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
  } break;

  case 6: {
    Cx6        images = input.readTensor<Cx6>();
    auto const fft = TOps::FFT<5, 3>(FirstN<5>(images.dimensions()));
    for (Index iv = 0; iv < images.dimension(5); iv++) {
      Cx5 img = images.chip<5>(iv);
      if (adj) {
        img = fft.adjoint(img);
      } else {
        img = fft.forward(img);
      }
      images.chip<5>(iv) = img;
    }
    writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Channels);
  } break;
  }
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
