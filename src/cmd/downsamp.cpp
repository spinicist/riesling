#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"

using namespace rl;

int main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> res(parser, "R", "Target resolution (4 mm)", {"res"}, 4.0);
  args::ValueFlag<Index> lores(parser, "L", "First N spokes are lo-res", {"lores"}, 0);
  args::Flag noShrink(parser, "S", "Do not shrink matrix", {"no-shrink"});
  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto traj = reader.trajectory();
  auto const [dsTraj, minRead] = traj.downsample(res.Get(), lores.Get(), !noShrink);
  auto const dsInfo = dsTraj.info();
  Cx4 ks = reader.readTensor<Cx4>(HD5::Keys::Noncartesian)
             .slice(Sz4{0, minRead, 0, 0}, Sz4{dsInfo.channels, dsInfo.read_points, dsInfo.spokes, dsInfo.volumes});

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "downsamp"));
  writer.writeTrajectory(dsTraj);
  writer.writeTensor(ks, HD5::Keys::Noncartesian);

  return EXIT_SUCCESS;
}