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

int main_cg(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts   GridOpts(parser);
  SDC::Opts   sdcOpts(parser, "pipe");
  SENSE::Opts senseOpts(parser);
  // args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> thr(parser, "T", "Termination threshold (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {"max-its"}, 8);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, GridOpts, traj, noncart), basis.dimension(0));
  auto const recon = make_recon(coreOpts, GridOpts, sdcOpts, traj, sense, basis);
  auto       normEqs = std::make_shared<NormalOp<Cx>>(recon);
  ConjugateGradients<Cx> cg{normEqs, its.Get(), thr.Get(), true};

  auto    sz = recon->ishape;
  Cropper out_cropper(LastN<3>(sz), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3     outSz = out_cropper.size();
  Cx5     out(sz[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residImage) { resid.resize(sz[0], outSz[0], outSz[1], outSz[2], nV); }

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < nV; iv++) {
    auto b = recon->adjoint(CChipMap(noncart, iv));
    auto x = cg.run(b.data());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm);
    if (coreOpts.residImage || coreOpts.residKSpace) { noncart.chip<4>(iv) -= recon->forward(xm); }
    if (coreOpts.residImage) {
      xm = recon->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm);
    }
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved(), resid, noncart);
  return EXIT_SUCCESS;
}
