#include "types.h"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/pad.hpp"
#include "parse_args.hpp"
#include "threads.hpp"

using namespace rl;

int main_pad(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<Sz3, Sz3Reader> padDims(parser, "SZ", "Pad/crop dimensions");
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
    Cx6 outImages(Sz6{inDims[0], inDims[1], padDims.Get()[0], padDims.Get()[1], padDims.Get()[2], inDims[5]});
    auto const start = Log::Now();
    if (fwd) {
      PadOp<5> pad(FirstN<5>(inDims), padDims.Get());
      for (Index ii = 0; ii < inDims[5]; ii++) {
        outImages.chip(ii, 5) = pad.forward(inImages.chip(ii, 5));
      }
      Log::Print(FMT_STRING("Pad took {}"), Log::ToNow(start));
    } else {
      PadOp<5> pad(AddFront(padDims.Get(), inDims[0], inDims[1]), MidN<2, 3>(inDims));
      for (Index ii = 0; ii < inDims[5]; ii++) {
        outImages.chip(ii, 5) = pad.adjoint(inImages.chip(ii, 5));
      }
      Log::Print(FMT_STRING("Pad Adjoint took {}"), Log::ToNow(start));
    }
    writer.writeTensor(outImages, HD5::Keys::Channels);
  } else {
    Cx5 inImages = reader.readTensor<Cx5>(HD5::Keys::Image);
    Sz5 inDims = inImages.dimensions();
    Cx5 outImages(Sz5{inDims[0], padDims.Get()[0], padDims.Get()[1], padDims.Get()[2], inDims[4]});
    auto const start = Log::Now();
    if (fwd) {
      PadOp<4> pad(FirstN<4>(inDims), padDims.Get());
      for (Index ii = 0; ii < inDims[4]; ii++) {
        outImages.chip(ii, 4) = pad.forward(inImages.chip(ii, 4));
      }
      Log::Print(FMT_STRING("Pad took {}"), Log::ToNow(start));
    } else {
      PadOp<4> pad(AddFront(padDims.Get(), inDims[0]), MidN<1, 3>(inDims));
      for (Index ii = 0; ii < inDims[4]; ii++) {
        outImages.chip(ii, 4) = pad.adjoint(inImages.chip(ii, 4));
      }
      Log::Print(FMT_STRING("Pad Adjoint took {}"), Log::ToNow(start));
    }
    writer.writeTensor(outImages, HD5::Keys::Image);
  }
  return EXIT_SUCCESS;
}
