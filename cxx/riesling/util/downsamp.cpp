#include "inputs.hpp"

#include "rl/filter.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

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
  auto const  ks1 = reader.readTensor<Cx5>();
  auto [dsTraj, ks2] = traj.downsample(ks1, res.Get(), trim, !noShrink, corners);
  float const M = *std::max_element(dsTraj.matrix().cbegin(), dsTraj.matrix().cend()) / 2.f;
  if (filter) {
    auto const f = filter.Get();
    NoncartesianTukey(f[0] * M, f[1] * M, f[2], dsTraj.points(), ks2);
  }

  HD5::Writer writer(oname.Get());
  dsTraj.write(writer);
  info.voxel_size = dsTraj.voxelSize();
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, ks2.dimensions(), ks2.data(), HD5::Dims::Noncartesian);
}
