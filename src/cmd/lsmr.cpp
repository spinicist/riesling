#include "types.hpp"

#include "algo/lsmr.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "op/tensorscale.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "scaling.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_lsmr(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index>       its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<float>       atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float>       btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float>       ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float>       λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, gridOpts, traj, noncart), basis.dimension(0));

  auto const A = make_recon(coreOpts, gridOpts, sdcOpts, traj, sense, basis);
  auto const M = make_kspace_pre(pre.Get(), A->oshape[0], traj, basis, preBias.Get(), coreOpts.ndft.Get());
  auto       debug = [&A](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), A->ishape, x.data());
  };
  LSMR lsmr{A, M, its.Get(), atol.Get(), btol.Get(), ctol.Get(), debug};

  auto const sz = A->ishape;
  Cropper    out_cropper(LastN<3>(sz), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3 const  outSz = out_cropper.size();
  Cx5        out(sz[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residImage) { resid.resize(sz[0], outSz[0], outSz[1], outSz[2], nV); }

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < nV; iv++) {
    auto x = lsmr.run(&noncart(0, 0, 0, 0, iv), λ.Get());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm);
    if (coreOpts.residImage || coreOpts.residKSpace) { noncart.chip<4>(iv) -= A->forward(xm); }
    if (coreOpts.residImage) {
      xm = A->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm);
    }
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved(), resid, noncart);
  Log::Print("Finished {}", parser.GetCommand().Name());
  return EXIT_SUCCESS;
}
