#include "types.hpp"

#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/wavelets.hpp"
#include "threads.hpp"

using namespace rl;

void main_wavelets(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> oname(parser, "FILE", "Output HD5 file");

  VectorFlag<Index>      dims(parser, "W", "Wavelet dimensions", {"dims"}, std::vector<Index>{1, 2, 3});
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  args::Flag adj(parser, "A", "Apply adjoint operation", {"adj"});
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }

  HD5::Reader    reader(iname.Get());
  auto           images = reader.readTensor<Cx5>();
  TOps::Wavelets wav(images.dimensions(), width.Get(), dims.Get());

  if (adj) {
    images = wav.adjoint(images);
  } else {
    images = wav.forward(images);
  }
  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader.readInfo());
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Image);
}
