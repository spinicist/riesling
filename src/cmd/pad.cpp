#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/pad.hpp"
#include "parse_args.h"
#include "threads.h"

using namespace rl;

int main_pad(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<Sz3, Sz3Reader> outSz(parser, "SZ", "Out size");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag channels(parser, "C", "Work on channels, not images", {'c', "channels"});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (image)", {'d', "dset"});
  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "pad", "h5"));
  writer.writeTrajectory(reader.trajectory());
  if (channels) {
    Cx6 inImages = reader.readTensor<Cx6>(HD5::Keys::Channels);
    Sz6 inDims = inImages.dimensions();
    Sz6 outDims{inDims[0], inDims[1], outSz.Get()[0], outSz.Get()[1], outSz.Get()[2], inDims[5]};
    Cx6 outImages(outDims);
    auto const start = Log::Now();
    if (fwd) {
      PadOp<6> pad(inDims, outDims);
      outImages = pad.A(inImages);
      Log::Print(FMT_STRING("Pad took {}"), Log::ToNow(start));
    } else {
      PadOp<6> pad(outDims, inDims);
      outImages = pad.Adj(inImages);
      Log::Print(FMT_STRING("Pad Adjoint took {}"), Log::ToNow(start));
    }
    writer.writeTensor(outImages, HD5::Keys::Channels);
  } else {
    Cx5 inImages = reader.readTensor<Cx5>(HD5::Keys::Image);
    Sz5 inDims = inImages.dimensions();
    Sz5 outDims{inDims[0], outSz.Get()[0], outSz.Get()[1], outSz.Get()[2], inDims[4]};
    Cx5 outImages(outDims);
    auto const start = Log::Now();
    if (fwd) {
      PadOp<5> pad(inDims, outDims);
      outImages = pad.A(inImages);
      Log::Print(FMT_STRING("Pad took {}"), Log::ToNow(start));
    } else {
      PadOp<5> pad(outDims, inDims);
      outImages = pad.Adj(inImages);
      Log::Print(FMT_STRING("Pad Adjoint took {}"), Log::ToNow(start));
    }
    writer.writeTensor(outImages, HD5::Keys::Image);
  }
  return EXIT_SUCCESS;
}
