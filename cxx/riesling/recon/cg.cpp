#include "types.hpp"

#include "algo/cg.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

void main_cg(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  SDC::Opts   sdcOpts(parser, "pipe");
  SENSE::Opts senseOpts(parser);
  // args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> thr(parser, "T", "Termination threshold (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {"max-its"}, 8);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, gridOpts, traj, noncart), basis.dimension(0));
  auto const recon = make_recon(coreOpts, gridOpts, sdcOpts, traj, sense, basis);
  auto       normEqs = std::make_shared<NormalOp<Cx>>(recon);
  ConjugateGradients<Cx> cg{normEqs, its.Get(), thr.Get(), true};

  auto    sz = recon->ishape;
  Cropper out_cropper(LastN<3>(sz), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3     outSz = out_cropper.size();
  Cx5     out(sz[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residual) { resid.resize(sz[0], outSz[0], outSz[1], outSz[2], nV); }
  for (Index iv = 0; iv < nV; iv++) {
    auto b = recon->adjoint(CChipMap(noncart, iv));
    auto x = cg.run(b.data());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm);
    if (coreOpts.residual) {
      noncart.chip<4>(iv) -= recon->forward(xm);
      xm = recon->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm);
    }
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  if (coreOpts.residual) {
    WriteOutput(coreOpts.residual.Get(), resid, info);
  }}
