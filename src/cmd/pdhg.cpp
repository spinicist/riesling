#include "types.hpp"

#include "algo/eig.hpp"
#include "algo/pdhg.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_pdhg(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index>       its(parser, "ITS", "Max iterations (4)", {"max-its"}, 4);

  args::ValueFlag<std::vector<float>, VectorReader<float>> σin(parser, "σ", "Pre-computed dual step sizes", {"sigma"});
  args::ValueFlag<float>                                   τin(parser, "τ", "Pre-computed primal step size", {"tau"}, -1.f);
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
  auto const shape = recon->ishape;
  auto       P = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  std::function<void(Index const, PDHG::Vector const &, PDHG::Vector const &, PDHG::Vector const &)> debug_x =
    [shape](Index const ii, PDHG::Vector const &x, PDHG::Vector const &x̅, PDHG::Vector const &xdiff) {
      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, x.data());
      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data());
      Log::Tensor(fmt::format("pdhg-xdiff-{:02d}", ii), shape, xdiff.data());
    };

  PDHG        pdhg(A, P, reg, σin.Get(), τin.Get(), debug_x);
  Cropper     out_cropper(LastN<3>(shape), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3         outSz = out_cropper.size();
  float const scale = Scaling(coreOpts.scaling, recon, P, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);
  Cx5 out(shape[0], outSz[0], outSz[1], outSz[2], nV);
  for (Index iv = 0; iv < nV; iv++) {
    auto x = pdhg.run(&noncart(0, 0, 0, 0, iv), its.Get());
    auto xm = Tensorfy(x, shape);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  return EXIT_SUCCESS;
}
