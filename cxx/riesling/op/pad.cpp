#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_pad(args::Subparser &parser)
{
  args::Positional<std::string>      iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string>      oname(parser, "FILE", "Output HD5 file");
  args::Positional<Sz3, SzReader<3>> padDims(parser, "SZ", "Pad/crop size on x,y,z dimensions");
  args::Flag                         fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));

  Cx6 const inImages = reader.readTensor<Cx6>();
  Sz6       inDims = inImages.dimensions();
  Sz6       outDims(inDims[0], inDims[1], padDims.Get()[0], padDims.Get()[1], padDims.Get()[2], inDims[5]);
  Cx6       outImages(outDims);

  if (fwd) {
    TOps::Pad<6> pad(inDims, outDims);
    outImages = pad.forward(inImages);
  } else {
    TOps::Pad<6> crop(outDims, inDims);
    outImages = crop.adjoint(inImages);
  }
  writer.writeTensor(HD5::Keys::Data, outImages.dimensions(), outImages.data(), HD5::Dims::Channels);
}
