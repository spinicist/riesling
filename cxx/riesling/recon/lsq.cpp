#include "types.hpp"

#include "algo/lsmr.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
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
  Index const nS = noncart.dimension(3);
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const A = Recon::SENSE(coreOpts, gridOpts, senseOpts, traj, nS, basis, noncart);
  auto const M =
    make_kspace_pre(traj, A->oshape[0], basis, gridOpts.vcc, preOpts.type.Get(), preOpts.bias.Get(), coreOpts.ndft.Get());
  auto debug = [&A](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), A->ishape, x.data(), {"v", "x", "y", "z"});
  };
  LSMR lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};

  TOps::Crop<Cx, 4> oc(A->ishape, AddFront(traj.matrixForFOV(coreOpts.fov.Get()), A->ishape[0]));
  Cx5               out(AddBack(oc.oshape, nV)), resid;
  if (coreOpts.residual) { resid.resize(out.dimensions()); }

  for (Index iv = 0; iv < nV; iv++) {
    auto x = lsmr.run(&noncart(0, 0, 0, 0, iv), lsqOpts.λ.Get());
    auto xm = Tensorfy(x, A->ishape);
    out.chip<4>(iv) = oc.forward(xm);
    if (coreOpts.residual) {
      noncart.chip<4>(iv) -= A->forward(xm);
      lsmr.iterLimit = 0;
      x = lsmr.run(&noncart(0, 0, 0, 0, iv), 0);
      lsmr.iterLimit = lsqOpts.its.Get();
      resid.chip<4>(iv) = oc.forward(xm);
    }    
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  if (coreOpts.residual) { WriteOutput(coreOpts.residual.Get(), resid, info); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
