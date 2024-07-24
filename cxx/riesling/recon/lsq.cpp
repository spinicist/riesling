#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "outputs.hpp"
#include "precon.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_recon_lsq(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  LsqOpts     lsqOpts(parser);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  Basis const basis(coreOpts.basisFile.Get());
  auto const A = Recon::SENSE(coreOpts.ndft, gridOpts, senseOpts, traj, nS, nT, &basis, noncart);
  auto const M = MakeKspacePre(traj, nC, nT, &basis, preOpts.type.Get(), preOpts.bias.Get(), coreOpts.ndft.Get());
  Log::Print("A {} {} M {} {}", A->ishape, A->oshape, M->rows(), M->cols());
  auto debug = [shape = A->ishape](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), shape, x.data(), HD5::Dims::Image);
  };
  LSMR lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};

  auto const x = lsmr.run(CollapseToConstVector(noncart), lsqOpts.λ.Get());
  auto const xm = Tensorfy(x, A->ishape);

  TOps::Crop<Cx, 5> oc(A->ishape, traj.matrixForFOV(coreOpts.fov.Get(), A->ishape[0], nT));
  auto              out = oc.forward(xm);
  basis.applyR(out);
  WriteOutput(coreOpts.oname.Get(), out, HD5::Dims::Image, info, Log::Saved());
  if (coreOpts.residual) {
    noncart -= A->forward(xm);
    Basis const id;
    auto const A1 = Recon::SENSE(coreOpts.ndft, gridOpts, senseOpts, traj, nS, nT, &id, noncart);
    auto const M1 =
      MakeKspacePre(traj, nC, nT, &id, preOpts.type.Get(), preOpts.bias.Get(), coreOpts.ndft.Get());
    Log::Print("A1 {} {} M1 {} {}", A1->ishape, A1->oshape, M1->rows(), M1->cols());
    Ops::Op<Cx>::Map  ncmap(noncart.data(), noncart.size());
    Ops::Op<Cx>::CMap nccmap(noncart.data(), noncart.size());
    M1->inverse(nccmap, ncmap);
    auto r = A1->adjoint(noncart);
    Log::Print("Finished calculating residual");
    HD5::Writer writer(coreOpts.residual.Get());
    writer.writeInfo(info);
    writer.writeTensor(HD5::Keys::Data, r.dimensions(), r.data(), HD5::Dims::Image);
  }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
