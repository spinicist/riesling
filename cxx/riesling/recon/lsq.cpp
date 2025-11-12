#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/log/debug.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/scaling.hpp"
#include "rl/sense/sense.hpp"
#include "rl/types.hpp"

using namespace rl;

template <int ND> void run_lsq(args::Subparser &parser)
{
  CoreArgs<ND>           coreArgs(parser);
  GridArgs<ND>           gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSEArgs<ND>          senseArgs(parser);
  LSMRArgs               lsqArgs(parser);
  f0Args                 f0Args(parser);
  ArrayFlag<float, ND>   cropFov(parser, "FOV", "Crop FoV in mm (x,y)", {"crop-fov"});
  args::ValueFlag<Index> debugIters(parser, "I", "Write debug images ever N iterations (1)", {"debug-iters"}, 1);

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const      cmd = parser.GetCommand().Name();
  HD5::Reader     reader(coreArgs.iname.Get());
  Info const      info = reader.readStruct<Info>(HD5::Keys::Info);
  TrajectoryN<ND> traj(reader, info.voxel_size.head<ND>(), coreArgs.matrix.Get());
  auto            noncart = reader.readTensor<Cx5>(coreArgs.dset.Get());
  traj.checkDims(FirstN<3>(noncart.dimensions()));

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const R = f0Args.Nτ ? Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, f0Args.Get(), noncart,
                                   reader.readTensor<Re3>("f0map"))
                           : Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);
  auto       debug = [shape = R.A->ishape, d = debugIters.Get()](Index const i, LSMR::Vector const &x, LSMR::Vector const &v) {
    if (i % d == 0) {
      Log::Tensor(fmt::format("lsmr-x-{:02d}", i), shape, x.data(), HD5::Dims::Images);
      Log::Tensor(fmt::format("lsmr-v-{:02d}", i), shape, v.data(), HD5::Dims::Images);
    }
  };
  LSMR lsmr{R.A, R.M, nullptr, lsqArgs.Get(), debug};

  auto const x = lsmr.run(CollapseToConstVector(noncart));
  auto const xm = AsTensorMap(x, R.A->ishape);

  TOps::Pad<5> oc(Concatenate(traj.matrixForFOV(cropFov.Get()), LastN<5 - ND>(R.A->ishape)), R.A->ishape);
  auto         out = oc.adjoint(xm);
  if (basis) { basis->applyR(out); }
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
  if (coreArgs.residual) {
    Log::Print(cmd, "Calculating K-space residual");
    fmt::print("|xm| {} |noncart| {}\n", Norm<true>(xm), Norm<true>(noncart));
    noncart -= R.A->forward(xm);
    fmt::print("|xm| {} |noncart| {}\n", Norm<true>(xm), Norm<true>(noncart));
    writer.writeTensor(HD5::Keys::Residual, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    traj.write(writer);
  }
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}

void main_recon_lsq(args::Subparser &parser) { run_lsq<3>(parser); }
void main_recon_lsq2(args::Subparser &parser) { run_lsq<2>(parser); }
