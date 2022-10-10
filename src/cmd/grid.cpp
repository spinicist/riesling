#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"
#include <filesystem>

using namespace rl;

int main_grid(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag bucket(parser, "", "Use bucket gridder", {"bucket"});

  ParseCommand(parser, coreOpts.iname);
  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto const info = traj.info();

  auto const basis = ReadBasis(coreOpts.basisFile);

  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "grid", "h5"));
  traj.write(writer);
  auto const start = Log::Now();
  if (fwd) {
    auto const cart = reader.readTensor<Cx5>(HD5::Keys::Cartesian);
    auto const gridder = make_grid<Cx, 3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), cart.dimension(0), basis);
    auto const rad_ks = gridder->forward(cart);
    writer.writeTensor(
      Cx5(rad_ks.reshape(Sz5{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1, 1})),
      HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Wrote non-cartesian k-space. Took {}"), Log::ToNow(start));
  } else {
    auto const noncart = reader.readSlab<Cx4>(HD5::Keys::Noncartesian, 0);
    Index const channels = noncart.dimension(0);
    auto const sdc = SDC::make_sdc(sdcOpts, traj, channels, coreOpts.ktype.Get(), coreOpts.osamp.Get());
    auto const gridder = make_grid<Cx, 3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), channels, basis);
    writer.writeTensor(gridder->adjoint(sdc->adjoint(noncart.chip<3>(0))), HD5::Keys::Cartesian);
    Log::Print(FMT_STRING("Wrote cartesian k-space. Took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
