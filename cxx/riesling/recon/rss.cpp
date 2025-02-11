#include "inputs.hpp"
#include "outputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/nufft.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_recon_rss(args::Subparser &parser)
{
  CoreArgs            coreArgs(parser);
  GridArgs<3>         gridArgs(parser);
  PreconArgs          preArgs(parser);
  LSMRArgs             lsqOpts(parser);
  ArrayFlag<float, 3> cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const A = TOps::NUFFTAll(gridArgs.Get(), traj, nC, nS, nT, basis.get());
  auto const M = MakeKSpaceSingle(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT);
  LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
  auto       x = lsmr.run(CollapseToConstVector(noncart));
  auto       xm = AsTensorMap(x, A->ishape);

  Cx5 const        rss = DimDot<1>(xm, xm).sqrt();
  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), A->ishape[0], nT), rss.dimensions());
  auto             out = oc.forward(rss);

  WriteOutput(cmd, coreArgs.oname.Get(), out, HD5::Dims::Image, info);
  Log::Print(cmd, "Finished");
}
