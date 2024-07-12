#include "types.hpp"

#include "filter.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

void main_filter(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");
  args::ValueFlag<float>        start(parser, "S", "Filter start in fractional k-space (default 0.5)", {"start"}, 0.5f);
  args::ValueFlag<float>        end(parser, "E", "Filter end in fractional k-space (default 1.0)", {"end"}, 1.f);
  args::ValueFlag<float>        height(parser, "H", "Filter end height (default 0.5)", {"height"}, 0.5f);
  ParseCommand(parser);

  if (!iname) { throw args::Error("No input file specified"); }
  HD5::Reader input(iname.Get());
  Cx5         images = input.readTensor<Cx5>();
  auto const  fft = TOps::FFT<4, 3>(FirstN<4>(images.dimensions()));
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    Cx4 img = images.chip<4>(iv);
    Cx4 ks = fft.forward(img);
    CartesianTukey(start.Get(), end.Get(), height.Get(), ks);
    img = fft.adjoint(ks);
    images.chip<4>(iv) = img;
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
