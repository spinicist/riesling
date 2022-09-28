#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_wavelets(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input image file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);
  args::ValueFlag<Index> levels(parser, "L", "Wavelet levels", {"levels", 'l'}, 4);
  args::Flag fwd(parser, "F", "Apply forward (encoding) operation", {"fwd"});
  ParseCommand(parser);

  if (!iname) {
    throw args::Error("No input file specified");
  }
  HD5::Reader reader(iname.Get());
  auto const input = reader.readTensor<Cx5>(HD5::Keys::Image);

  Wavelets wavelets(FirstN<4>(input.dimensions()), width.Get(), levels.Get());
  Cx5 output(input.dimensions());
  auto const start = Log::Now();
  for (Index iv = 0; iv < input.dimension(4); iv++) {
    if (fwd) {
      output.chip<4>(iv) = wavelets.forward(input.chip<4>(iv));
    } else {
      output.chip<4>(iv) = wavelets.adjoint(input.chip<4>(iv));
    }
  }
  Log::Print(FMT_STRING("All volumes took {}"), Log::ToNow(start));
  auto const fname = OutName(iname.Get(), oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(output, HD5::Keys::Image);

  return EXIT_SUCCESS;
}
