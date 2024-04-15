#include "types.hpp"

#include "filter.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string>                  iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string>                   oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Eigen::Array3f, Array3fReader> res(parser, "R", "Target resolution (4 mm)", {"res"},
                                                     Eigen::Array3f::Constant(4.f));
  args::ValueFlag<Index>                         lores(parser, "L", "First N traces are lo-res", {"lores"}, 0);
  args::ValueFlag<float>                         filterStart(parser, "T", "Tukey filter start", {"filter-start"}, 0.5f);
  args::ValueFlag<float>                         filterEnd(parser, "T", "Tukey filter end", {"filter-end"}, 1.0f);
  args::Flag                                     noShrink(parser, "S", "Do not shrink matrix", {"no-shrink"});
  args::Flag                                     corners(parser, "C", "Keep corners", {"corners"});
  ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get());
  Info        info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const  ks1 = reader.readTensor<Cx5>();
  auto [dsTraj, ks2] = traj.downsample(ks1, res.Get(), lores.Get(), !noShrink, corners);

  if (filterStart || filterEnd) {
    NoncartesianTukey(filterStart.Get() * 0.5, filterEnd.Get() * 0.5, 0.f, dsTraj.points(), ks2);
  }

  HD5::Writer writer(oname.Get());
  dsTraj.write(writer);
  info.voxel_size = dsTraj.voxelSize();
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, ks2.dimensions(), ks2.data());

  return EXIT_SUCCESS;
}
