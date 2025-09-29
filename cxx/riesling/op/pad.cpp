#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

template <int R> void DoPad(HD5::Reader &reader, HD5::Writer &writer, Sz3 const shape, bool const fwd)
{
  CxN<R> const inImages = reader.readTensor<CxN<R>>();
  Sz<R>       inDims = inImages.dimensions();
  Sz<R>       outDims;
  for (Index ii = 0; ii < 3; ii++) {
    outDims[ii] = shape[ii];
  }
  for (Index ii = 3; ii < R; ii++) {
    outDims[ii] = inDims[ii];
  }
  CxN<R> outImages(outDims);

  if (fwd) {
    TOps::Pad<R> pad(inDims, outDims);
    outImages = pad.forward(inImages);
  } else {
    TOps::Pad<R> crop(outDims, inDims);
    outImages = crop.adjoint(inImages);
  }
  writer.writeTensor(HD5::Keys::Data, outImages.dimensions(), outImages.data(), reader.readDNames<R>());
}

void main_pad(args::Subparser &parser)
{
  args::Positional<std::string>      iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string>      oname(parser, "FILE", "Output HD5 file");
  args::Positional<Sz3, SzReader<3>> padDims(parser, "SZ", "Pad/crop size on x,y,z dimensions");
  args::Flag                         fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  HD5::Writer writer(oname.Get());
  if (reader.exists(HD5::Keys::Info)) { writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info)); }
  switch (reader.order()) {
  case 4: DoPad<4>(reader, writer, padDims.Get(), fwd); break;
  case 5: DoPad<5>(reader, writer, padDims.Get(), fwd); break;
  case 6: DoPad<6>(reader, writer, padDims.Get(), fwd); break;
  default: throw(Log::Failure("pad", "Unimplemented order {}", reader.order()));
  }
}
