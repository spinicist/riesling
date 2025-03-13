#include "types.hpp"

#include "algo/lad.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "outputs.hpp"
#include "precon.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_recon_lad(args::Subparser &parser)
{
  CoreArgs            coreArgs(parser);
  GridArgs<3>         gridArgs(parser);
  PreconArgs          preArgs(parser);
  ReconArgs           reconArgs(parser);
  SENSE::Opts         senseOpts(parser);
  ArrayFlag<float, 3> cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());

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

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const A = Recon::Choose(reconArgs.Get(), gridArgs.Get(), senseOpts, traj, basis.get(), noncart);
  auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT, basis.get());

  LAD lad{A,       M,       inner_its0.Get(), inner_its1.Get(), atol.Get(), btol.Get(), ctol.Get(), outer_its.Get(),
          ε.Get(), μ.Get(), τ.Get()};

  auto const x = lad.run(noncart.data(), ρ.Get());
  auto const xm = AsTensorMap(x, A->ishape);

  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), A->ishape[3], nT), A->ishape);
  auto             out = oc.adjoint(xm);
  WriteOutput(cmd, coreArgs.oname.Get(), out, HD5::Dims::Images, info);
  if (coreArgs.residual) {
    WriteResidual(cmd, coreArgs.oname.Get(), reconArgs.Get(), gridArgs.Get(), senseOpts, preArgs.Get(), traj, xm, A, noncart);
  }
  Log::Print(cmd, "Finished");
}
