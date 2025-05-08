#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/wavelets.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

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
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
  writer.writeTensor(HD5::Keys::Data, images.dimensions(), images.data(), HD5::Dims::Images);
}
