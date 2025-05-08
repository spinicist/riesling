#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_merge(args::Subparser &parser)
{
  args::Positional<std::string> iname1(parser, "F", "First file");
  args::Positional<std::string> iname2(parser, "F", "Second file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file");

  args::ValueFlag<float> scale1(parser, "S", "Scale 1", {"scale1"}, 1.f);
  args::ValueFlag<float> scale2(parser, "S", "Scale 2", {"scale2"}, 1.f);

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!iname1) { throw args::Error("Input file 1 not specified"); }
  if (!iname2) { throw args::Error("Input file 2 not specified"); }
  if (!oname) { throw args::Error("Output file not specified"); }

  HD5::Reader reader1(iname1.Get());
  HD5::Reader reader2(iname2.Get());
  Trajectory  traj1(reader1, reader1.readStruct<Info>(HD5::Keys::Info).voxel_size);
  Trajectory  traj2(reader2, reader2.readStruct<Info>(HD5::Keys::Info).voxel_size);

  if (!traj1.compatible(traj2)) { throw Log::Failure(cmd, "Trajectories are not compatible"); }

  Cx5 ks1 = reader1.readTensor<Cx5>();
  Cx5 ks2 = reader2.readTensor<Cx5>();

  if (scale1) { ks1 = ks1 * ks1.constant(scale1.Get()); }
  if (scale2) { ks2 = ks2 * ks2.constant(scale2.Get()); }

  Index const nC1 = ks1.dimension(0);
  Index const nS1 = ks1.dimension(1);
  Index const nT1 = ks1.dimension(2);
  Index const nSl1 = ks1.dimension(3);
  Index const nV1 = ks1.dimension(4);
  Index const nC2 = ks2.dimension(0);
  Index const nS2 = ks2.dimension(1);
  Index const nT2 = ks2.dimension(2);
  Index const nSl2 = ks2.dimension(3);
  Index const nV2 = ks2.dimension(4);
  if (nC1 != nC2) { throw Log::Failure(cmd, "Datasets have {} and {} channels", nC1, nC2); }
  if (nSl1 != nSl2) { throw Log::Failure(cmd, "Datasets have {} and {} slabs", nSl1, nSl2); }
  if (nV1 != nV2) { throw Log::Failure(cmd, "Datasets have {} and {} volumes", nV1, nV2); }

  if (nS1 != nS2) {
    Log::Print(cmd, "Datasets have unequal samples ({}, {}), pruning to shortest", nS1, nS2);
    Index const minS = std::min(nS1, nS2);
    ks1 = Cx5(ks1.slice(Sz5{}, Sz5{nC1, minS, nT1, nSl1, nV1}));
    traj1 = Trajectory(traj1.points().slice(Sz3{}, Sz3{3, minS, nT1}), traj1.matrix(), traj1.voxelSize());
    ks2 = Cx5(ks2.slice(Sz5{}, Sz5{nC2, minS, nT2, nSl2, nV2}));
    traj2 = Trajectory(traj2.points().slice(Sz3{}, Sz3{3, minS, nT2}), traj2.matrix(), traj2.voxelSize());
  }

  Log::Print(cmd, "Merging across traces");
  Trajectory traj(traj1.points().concatenate(traj2.points(), 2), traj1.matrix(), traj1.voxelSize());
  Cx5 const  ks = ks1.concatenate(ks2, 2);

  HD5::Writer writer(oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader1.readStruct<Info>(HD5::Keys::Info));
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
  Log::Print(cmd, "Finshed");
}
