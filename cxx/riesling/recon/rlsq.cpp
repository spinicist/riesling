#include "inputs.hpp"
#include "outputs.hpp"
#include "regularizers.hpp"

#include "rl/algo/admm.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/scaling.hpp"
#include "rl/sense/sense.hpp"

using namespace rl;

void main_recon_rlsq(args::Subparser &parser)
{
  CoreArgs<3>                  coreArgs(parser);
  GridArgs<3>                  gridArgs(parser);
  PreconArgs                   preArgs(parser);
  ReconArgs                    reconArgs(parser);
  SENSEArgs                    senseArgs(parser);
  ADMMArgs                     admmArgs(parser);
  RegOpts                      regOpts(parser);
  f0Args                       f0Args(parser);
  args::ValueFlag<std::string> scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu");
  args::ValueFlag<Index>       debugIters(parser, "I", "Write debug images ever N outer iterations (16)", {"debug-iters"}, 16);
  args::Flag                   debugZ(parser, "Z", "Write regularizer debug images", {"debug-z"});
  ArrayFlag<float, 3>          cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const R = f0Args.Nτ ? Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, f0Args.Get(), noncart,
                                   reader.readTensor<Re3>("f0map"))
                           : Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);
  auto const shape = R.A->ishape;
  float const scale = ScaleData(scaling.Get(), R.A, R.M, CollapseToVector(noncart));
  if (scale != 1.f) { noncart.device(Threads::TensorDevice()) = noncart * Cx(scale); }
  auto [reg, A, ext_x] = Regularizers(regOpts, R.A);

  ADMM::DebugX debug_x = [shape, di = debugIters.Get()](Index const ii, ADMM::Vector const &x) {
    if (ii % di == 0) { Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data(), HD5::Dims::Images); }
  };

  ADMM::DebugZ debug_z = [&, di = debugIters.Get()](Index const ii, Index const ir, ADMM::Vector const &Fx,
                                                    ADMM::Vector const &z, ADMM::Vector const &u) {
    if (debugZ && (ii % di == 0)) {
      if (std::holds_alternative<Sz5>(reg[ir].shape)) {
        auto const Fshape = std::get<Sz5>(reg[ir].shape);
        Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), HD5::Dims::Images);
        Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), HD5::Dims::Images);
        Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), HD5::Dims::Images);
      }
      if (std::holds_alternative<Sz6>(reg[ir].shape)) {
        auto const Fshape = std::get<Sz6>(reg[ir].shape);
        Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), {"b", "i", "j", "k", "t", "g"});
        Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), {"b", "i", "j", "k", "t", "g"});
        Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), {"b", "i", "j", "k", "t", "g"});
      }
    }
  };

  ADMM opt{A, R.M, reg, admmArgs.Get(), debug_x, debug_z};

  auto x = ext_x ? ext_x->forward(opt.run(CollapseToConstVector(noncart))) : opt.run(CollapseToConstVector(noncart));
  if (scale != 1.f) { x.device(Threads::CoreDevice()) = x / Cx(scale); }
  auto const xm = AsConstTensorMap(x, R.A->ishape);

  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), shape[3], shape[4]), R.A->ishape);
  auto             out = oc.adjoint(xm);
  if (basis) { basis->applyR(out); }
  WriteOutput(cmd, coreArgs.oname.Get(), out, HD5::Dims::Images, info);
  if (coreArgs.residual) {
    WriteResidual(cmd, coreArgs.oname.Get(), reconArgs.Get(), gridArgs.Get(), senseArgs.Get(), preArgs.Get(), traj, xm, R.A,
                  noncart);
  }
  Log::Print(cmd, "Finished");
}
