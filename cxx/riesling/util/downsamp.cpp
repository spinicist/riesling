#include "inputs.hpp"

#include "rl/filter.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include <flux.hpp>

using namespace rl;

void main_downsamp(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT", "Input file name");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file name");
  ArrayFlag<float, 3>           res(parser, "R", "Target resolution (4 mm)", {"res"}, Eigen::Array3f::Constant(4.f));
  ArrayFlag<float, 3>           filter(parser, "F", "Filter start,end,height", {'f', "filter"});
  args::Flag                    noShrink(parser, "S", "Do not shrink matrix", {"no-shrink"});
  args::Flag                    trim(parser, "T", "Trim non-cartesian", {"trim"});
  args::Flag                    corners(parser, "C", "Keep corners", {"corners"});
  ParseCommand(parser, iname, oname);

  HD5::Reader reader(iname.Get());
  Info        info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  Cx5  ks1 = reader.readTensor<Cx5>();
  traj.downsample(res.Get(), !noShrink, corners);
  if (trim) { ks1 = Cx5(traj.trim(ks1)); }
  if (filter) {
    auto const f = filter.Get();
    auto const M = *flux::max(traj.matrix()) / 2.f;
    NoncartesianTukey(f[0] * M, f[1] * M, f[2], traj.points(), ks1);
  }

  HD5::Writer writer(oname.Get());
  traj.write(writer);
  info.voxel_size = traj.voxelSize();
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, ks1.dimensions(), ks1.data(), HD5::Dims::Noncartesian);
}
