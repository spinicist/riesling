#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"
#include "filter.hpp"

using namespace rl;

int main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<float> res(parser, "R", "Target resolution (4 mm)", {"res"}, 4.0);
  args::ValueFlag<Index> lores(parser, "L", "First N traces are lo-res", {"lores"}, 0);
  args::ValueFlag<float> filterStart(parser, "T", "Tukey filter start", {"filter-start"}, 0.5f);
  args::ValueFlag<float> filterEnd(parser, "T", "Tukey filter end", {"filter-end"}, 1.0f);
  args::Flag noShrink(parser, "S", "Do not shrink matrix", {"no-shrink"});
  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Trajectory traj(reader);
  auto const ks1 = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  auto [dsTraj, ks2] = traj.downsample(ks1, res.Get(), lores.Get(), !noShrink);

  if (filterStart || filterEnd) {
    NoncartesianTukey(filterStart.Get() * 0.5, filterEnd.Get()*0.5, 0.f, dsTraj.points(), ks2);
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), parser.GetCommand().Name()));
  dsTraj.write(writer);
  writer.writeTensor(HD5::Keys::Noncartesian, ks2.dimensions(), ks2.data());

  return EXIT_SUCCESS;
}
