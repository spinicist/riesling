#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/gridBase.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"
#include <filesystem>

using namespace rl;

int main_grid(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag bucket(parser, "", "Use bucket gridder", {"bucket"});

  ParseCommand(parser, core.iname);
  HD5::Reader reader(core.iname.Get());
  Trajectory traj(reader);
  auto const info = traj.info();

  auto const basis = ReadBasis(core.basisFile);

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "grid", "h5"));
  writer.writeTrajectory(traj);
  auto const start = Log::Now();
  if (fwd) {
    auto const cart = reader.readTensor<Cx5>(HD5::Keys::Cartesian);
    auto const gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), cart.dimension(0), basis);
    auto const rad_ks = gridder->forward(cart);
    writer.writeTensor(
      Cx4(rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1})),
      HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Wrote non-cartesian k-space. Took {}"), Log::ToNow(start));
  } else {
    auto const noncart = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, 0);
    Index const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
    auto const gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
    writer.writeTensor(gridder->adjoint(sdc->adjoint(noncart)), HD5::Keys::Cartesian);
    Log::Print(FMT_STRING("Wrote cartesian k-space. Took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
