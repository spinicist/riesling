#include "types.h"

#include "filter.hpp"
#include "io/hd5.hpp"
#include "log.h"
#include "op/fft.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_filter(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Basis images file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> start(parser, "S", "Filter start in fractional k-space (default 0.5)", {"start"}, 0.5f);
  args::ValueFlag<float> end(parser, "E", "Filter end in fractional k-space (default 1.0)", {"end"}, 1.f);
  args::ValueFlag<float> height(parser, "H", "Filter end height (default 0.5)", {"height"}, 0.5f);
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader input(iname.Get());
  Cx5 images = input.readTensor<Cx5>(HD5::Keys::Image);
  auto const fft = FFT::Make<4, 3>(FirstN<4>(images.dimensions()));
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Cx4 img = images.chip<4>(iv);
    fft->forward(img);
    KSTukey(start.Get(), end.Get(), height.Get(), img);
    fft->reverse(img);
    images.chip<4>(iv) = img;
  }
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(input.readInfo());
  writer.writeTensor(images, HD5::Keys::Image);
  return EXIT_SUCCESS;
}
