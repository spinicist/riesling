#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> res(parser, "R", "Target resolution (4 mm)", {"res"}, 4.0);
  args::ValueFlag<Index> lores(parser, "L", "First N traces are lo-res", {"lores"}, 0);
  args::Flag noShrink(parser, "S", "Do not shrink matrix", {"no-shrink"});
  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Trajectory traj(reader);
  Cx4 ks1 = reader.readTensor<Cx4>(HD5::Keys::Noncartesian);
  auto const [dsTraj, ks2] = traj.downsample(ks1, res.Get(), lores.Get(), !noShrink);
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "downsamp"));
  dsTraj.write(writer);
  writer.writeTensor(ks2, HD5::Keys::Noncartesian);

  return EXIT_SUCCESS;
}