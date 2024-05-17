#include "types.hpp"

#include "algo/lad.hpp"
#include "cropper.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_recon_lad(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<Index> inner_its0(parser, "ITS", "Initial inner iterations (4)", {"max-its0"}, 4);
  args::ValueFlag<Index> inner_its1(parser, "ITS", "Subsequenct inner iterations (1)", {"max-its"}, 1);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (20)", {"max-outer-its"}, 20);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f);
  args::ValueFlag<float> ε(parser, "ε", "ADMM convergence tolerance (1e-2)", {"eps"}, 1.e-2f);
  args::ValueFlag<float> μ(parser, "μ", "ADMM residual rescaling tolerance (default 1.2)", {"mu"}, 1.2f);
  args::ValueFlag<float> τ(parser, "τ", "ADMM residual rescaling maximum (default 10)", {"tau"}, 10.f);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nS = noncart.dimension(3);
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, noncart), basis.dimension(0));
  auto const A = Recon::SENSE(coreOpts, gridOpts, traj, nS, sense, basis);
  auto const shape = A->ishape;
  auto const M = make_kspace_pre(traj, A->oshape[0], basis, preOpts.type.Get(), preOpts.bias.Get());

  Cropper out_cropper(LastN<3>(shape), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3     outSz = out_cropper.size();

  LAD lad{A,       M,       inner_its0.Get(), inner_its1.Get(), atol.Get(), btol.Get(), ctol.Get(), outer_its.Get(),
          ε.Get(), μ.Get(), τ.Get()};

  Cx5 out(shape[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residual) { resid.resize(shape[0], outSz[0], outSz[1], outSz[2], nV); }

  for (Index iv = 0; iv < nV; iv++) {
    auto x = lad.run(&noncart(0, 0, 0, 0, iv), ρ.Get());
    auto xm = Tensorfy(x, shape);
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
