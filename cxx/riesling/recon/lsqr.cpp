#include "types.hpp"

#include "algo/lsqr.hpp"
#include "cropper.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "scaling.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"
#include "tensors.hpp"
#include "threads.hpp"

using namespace rl;

void main_lsqr(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<Index>       its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<float>       atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float>       btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float>       ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float>       λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, noncart), basis.dimension(0));
  auto const A = SENSERecon(coreOpts, gridOpts, sdcOpts, traj, sense, basis);
  auto const M = make_kspace_pre(pre.Get(), A->oshape[0], traj, basis, preBias.Get());
  auto       debug = [&A](Index const i, LSQR::Vector const &x) {
    Log::Tensor(fmt::format("lsqr-x-{:02d}", i), A->ishape, x.data());
  };
  LSQR lsqr{A, M, its.Get(), atol.Get(), btol.Get(), ctol.Get(), debug};

  auto      sz = A->ishape;
  Cropper   out_cropper(LastN<3>(sz), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3 const outSz = out_cropper.size();
  Cx5       out(sz[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residual) { resid.resize(sz[0], outSz[0], outSz[1], outSz[2], nV); }
  for (Index iv = 0; iv < nV; iv++) {
    auto x = lsqr.run(&noncart(0, 0, 0, 0, iv), λ.Get());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm);
    if (coreOpts.residual) {
      noncart.chip<4>(iv) -= A->forward(xm);
      xm = A->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm);
    }
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  if (coreOpts.residual) { WriteOutput(coreOpts.residual.Get(), resid, info); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
