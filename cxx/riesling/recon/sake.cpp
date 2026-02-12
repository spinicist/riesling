#include "inputs.hpp"

#include "rl/algo/pdhg.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/debug.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/loopify.hpp"
#include "rl/op/nufft.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/prox/llr.hpp"
#include "rl/scaling.hpp"

using namespace rl;

void main_recon_rrss(args::Subparser &parser)
{
  CoreArgs<3>                  coreArgs(parser);
  GridArgs<3>                  gridArgs(parser);
  PreconArgs                   preArgs(parser);
  PDHGArgs                     pdhgArgs(parser);
  ArrayFlag<float, 3>          cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());
  args::ValueFlag<float>       λ(parser, "L", "Regularization parameter (default 1e-1)", {"lambda"}, 1.e-1f);
  args::ValueFlag<std::string> scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu");
  args::ValueFlag<Index>       restart(parser, "R", "Restart PDHG", {"pdhg-restart", 'r'}, 8);
  args::ValueFlag<Index>       debugIters(parser, "I", "Write debug images ever N outer iterations (16)", {"debug-iters"}, 16);

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

  auto const  nufft = TOps::MakeNUFFT<3>(gridArgs.Get(), traj, nC, basis.get());
  auto const  A = Loopify<3>(nufft, nS, nT);
  auto const  M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, basis.get(), nC, Sz2{nS, nT});
  Sz6 const   shape = A->ishape;
  float const scale = ScaleData(scaling.Get(), A, M, CollapseToVector(noncart));
  if (scale != 1.f) { noncart.device(Threads::TensorDevice()) = noncart * Cx(scale); }

  std::vector<Regularizer> regs;
  regs.push_back({nullptr, std::make_shared<Proxs::LLR<6>>(λ.Get(), 8, 8, true, shape), shape});

  PDHG::Debug debug = [shape, di = debugIters.Get()](Index const ii, PDHG::Vector const &dx, PDHG::Vector const &x̅) {
    if (Log::IsDebugging() && (ii % di == 0)) {
      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, dx.data(), HD5::Dims::Channels);
      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data(), HD5::Dims::Channels);
    }
  };
  Ops::Op::Vector x = restart
                        ? PDHG::Restarted(restart.Get(), CollapseToConstVector(noncart), A, M, regs, pdhgArgs.Get(), debug)
                        : PDHG::Run(CollapseToConstVector(noncart), A, M, regs, pdhgArgs.Get(), debug);

  auto         xm = AsTensorMap(x, A->ishape);
  Cx5 const    rss = DimDot<4>(xm, xm).sqrt();
  TOps::Pad<5> oc(traj.matrixForFOV(cropFov.Get(), rss.dimension(3), nT), rss.dimensions());
  auto         out = oc.adjoint(rss);
  HD5::Writer  writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
  if (coreArgs.residual) { Log::Warn(cmd, "RSS does not support residual output"); }
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}
