#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "inputs.hpp"
#include "threads.hpp"

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
  writer.writeInfo(reader.readInfo());

  Cx6 const  inImages = reader.readTensor<Cx6>();
  Sz6        inDims = inImages.dimensions();
  Sz6        outDims(inDims[0], inDims[1], padDims.Get()[0], padDims.Get()[1], padDims.Get()[2], inDims[5]);
  Cx6        outImages(outDims);

  if (fwd) {
    TOps::Pad<Cx, 6> pad(inDims, outDims);
    outImages = pad.forward(inImages);
  } else {
    TOps::Crop<Cx, 6> crop(inDims, outDims);
    outImages = crop.forward(inImages);
  }
  writer.writeTensor(HD5::Keys::Data, outImages.dimensions(), outImages.data(), HD5::Dims::Channels);
}

