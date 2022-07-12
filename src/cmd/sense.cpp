#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/sense.hpp"
#include "parse_args.h"

using namespace rl;

int main_sense(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::Positional<std::string> sname(parser, "FILE", "SENSE maps HD5 file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (image/channels)", {'d', "dset"});
  ParseCommand(parser, iname);
  HD5::RieslingReader ireader(iname.Get());
  if (!sname) {
    Log::Fail(FMT_STRING("No input SENSE map file specified"));
  }
  HD5::Reader sreader(sname.Get());
  auto const maps = sreader.readTensor<Cx4>(HD5::Keys::SENSE);

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "sense"));
  writer.writeTrajectory(ireader.trajectory());

  auto const start = Log::Now();
  if (fwd) {
    std::string const name = dset ? dset.Get() : HD5::Keys::Image;
    auto const images = ireader.readTensor<Cx5>(name);
    if (!std::equal(images.dimensions().begin() + 1, images.dimensions().begin() + 4, maps.dimensions().begin() + 1)) {
      Log::Fail(
        FMT_STRING("Image dimensions {} did not match SENSE maps {}"),
        fmt::join(images.dimensions(), ","),
        fmt::join(maps.dimensions(), ","));
    }
    SenseOp sense(maps, images.dimension(0));
    Cx6 channels(
      maps.dimension(0),
      images.dimension(0),
      maps.dimension(1),
      maps.dimension(2),
      maps.dimension(3),
      images.dimension(4));
    for (auto ii = 0; ii < images.dimension(4); ii++) {
      channels.chip<5>(ii).device(Threads::GlobalDevice()) = sense.A(images.chip<4>(ii));
    }
    Log::Print(FMT_STRING("SENSE took {}"), Log::ToNow(start));
    writer.writeTensor(channels, HD5::Keys::Channels);
  } else {
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    auto const channels = ireader.readTensor<Cx6>(name);
    if (channels.dimension(0) != maps.dimension(0)) {
      Log::Fail(
        FMT_STRING("Number of channels {} did not match SENSE maps {}"), channels.dimension(0), maps.dimension(0));
    }
    if (!std::equal(channels.dimensions().begin() + 2, channels.dimensions().end(), maps.dimensions().begin() + 1)) {
      Log::Fail(
        FMT_STRING("Image dimensions {} did not match SENSE maps {}"),
        fmt::join(channels.dimensions(), ","),
        fmt::join(maps.dimensions(), ","));
    }
    SenseOp sense(maps, channels.dimension(1));
    Cx5 images(LastN<5>(channels.dimensions()));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      images.chip<4>(ii).device(Threads::GlobalDevice()) = sense.Adj(channels.chip<5>(ii));
    }
    Log::Print(FMT_STRING("SENSE Adjoint took {}"), Log::ToNow(start));
    writer.writeTensor(images, HD5::Keys::Image);
  }

  return EXIT_SUCCESS;
}
