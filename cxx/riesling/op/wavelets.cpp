#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

void main_wavelets(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  args::ValueFlag<Sz4, SzReader<4>> dims(parser, "W", "Wavelet dimensions", {"dims"}, Sz4{0, 1, 1, 1});
  args::ValueFlag<Index>            width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  args::Flag adj(parser, "A", "Apply adjoint operation", {"adj"});
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader reader(iname.Get());
  auto        images = reader.readTensor<Cx5>();
  TOps::Wavelets    wav(FirstN<4>(images.dimensions()), width.Get(), dims.Get());
  for (Index iv = 0; iv < images.dimension(4); iv++) {
    if (adj) {
      images.chip<4>(iv) = wav.adjoint(ChipMap(images, iv));
    } else {
      images.chip<4>(iv) = wav.forward(ChipMap(images, iv));
    }
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
}
