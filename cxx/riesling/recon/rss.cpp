#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/loopify.hpp"
#include "rl/op/nufft.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_recon_rss(args::Subparser &parser)
{
  CoreArgs<3>         coreArgs(parser);
  GridArgs<3>         gridArgs(parser);
  PreconArgs          preArgs(parser);
  LSMRArgs            lsqOpts(parser);
  ArrayFlag<float, 3> cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readStruct<Info>(HD5::Keys::Info);
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>(coreArgs.dset.Get());
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const nufft = TOps::MakeNUFFT<3>(gridArgs.Get(), traj, nC, basis.get());
  auto const A = Loopify<3>(nufft, nS, nT);
  auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, nC, Sz2{nS, nT});
  LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
  auto       x = lsmr.run(CollapseToConstVector(noncart));
  auto       xm = AsTensorMap(x, A->ishape);

  Cx5 const    rss = DimDot<3>(xm, xm).sqrt();
  TOps::Pad<5> oc(traj.matrixForFOV(cropFov.Get(), rss.dimension(3), nT), rss.dimensions());
  auto         out = oc.adjoint(rss);
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);  
  if (coreArgs.residual) { Log::Warn(cmd, "RSS does not support residual output"); }
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}
