#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "inputs.hpp"
#include "tensors.hpp"

using namespace rl;

void main_merge(args::Subparser &parser)
{
  args::Positional<std::string> iname1(parser, "F", "First file");
  args::Positional<std::string> iname2(parser, "F", "Second file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file");

  args::ValueFlag<float> scale1(parser, "S", "Scale 1", {"scale1"}, 1.f);
  args::ValueFlag<float> scale2(parser, "S", "Scale 2", {"scale2"}, 1.f);

  ParseCommand(parser);
  if (!iname1) { throw args::Error("Input file 1 not specified"); }
  if (!iname2) { throw args::Error("Input file 2 not specified"); }
  if (!oname) { throw args::Error("Output file not specified"); }

  HD5::Reader reader1(iname1.Get());
  HD5::Reader reader2(iname2.Get());
  Trajectory  traj1(reader1, reader1.readInfo().voxel_size);
  Trajectory  traj2(reader2, reader2.readInfo().voxel_size);

  if (!traj1.compatible(traj2)) { Log::Fail("Trajectories are not compatible"); }

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
  if (nC1 != nC2) { Log::Fail("Datasets have {} and {} channels", nC1, nC2); }
  if (nSl1 != nSl2) { Log::Fail("Datasets have {} and {} slabs", nSl1, nSl2); }
  if (nV1 != nV2) { Log::Fail("Datasets have {} and {} volumes", nV1, nV2); }

  if (nS1 != nS2) {
    Log::Print("Datasets have unequal samples ({}, {}), pruning to shortest", nS1, nS2);
    Index const minS = std::min(nS1, nS2);
    ks1 = Cx5(ks1.slice(Sz5{}, Sz5{nC1, minS, nT1, nSl1, nV1}));
    traj1 = Trajectory(traj1.points().slice(Sz3{}, Sz3{3, minS, nT1}), traj1.matrix(), traj1.voxelSize());
    ks2 = Cx5(ks2.slice(Sz5{}, Sz5{nC2, minS, nT2, nSl2, nV2}));
    traj2 = Trajectory(traj2.points().slice(Sz3{}, Sz3{3, minS, nT2}), traj2.matrix(), traj2.voxelSize());
  }

  Log::Print("Merging across traces");
  Trajectory traj(traj1.points().concatenate(traj2.points(), 2), traj1.matrix(), traj1.voxelSize());
  Cx5 const  ks = ks1.concatenate(ks2, 2);
  Log::Print("t1 {} ks1 {} t2 {} ks2 {} k {} ks {}", traj1.points().dimensions(), ks1.dimensions(), traj2.points().dimensions(),
             ks2.dimensions(), traj.points().dimensions(), ks.dimensions());

  HD5::Writer writer(oname.Get());
  writer.writeInfo(reader1.readInfo());
  traj.write(writer);
  writer.writeTensor(HD5::Keys::Data, ks.dimensions(), ks.data(), HD5::Dims::Noncartesian);
}
