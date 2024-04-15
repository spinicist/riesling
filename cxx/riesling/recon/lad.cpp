#include "types.hpp"

#include "algo/lad.hpp"
#include "algo/lsqr.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "scaling.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_lad(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index>       inner_its(parser, "ITS", "Max inner iterations (4)", {"max-its"}, 4);
  args::ValueFlag<float>       atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float>       btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float>       ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (8)", {"max-outer-its"}, 8);
  args::ValueFlag<float> abstol(parser, "ABS", "ADMM absolute tolerance (1e-3)", {"abs-tol"}, 1.e-4f);
  args::ValueFlag<float> reltol(parser, "REL", "ADMM relative tolerance (1e-3)", {"rel-tol"}, 1.e-3f);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM penalty parameter ρ (default 1)", {"rho"}, 1.f);
  args::ValueFlag<float> α(parser, "α", "ADMM relaxation α (default 1)", {"relax"}, 1.f);
  args::ValueFlag<float> μ(parser, "μ", "ADMM primal-dual mismatch limit (10)", {"mu"}, 10.f);
  args::ValueFlag<float> τ(parser, "τ", "ADMM primal-dual rescale (2)", {"tau"}, 2.f);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  Info const &info = traj.info();
  auto        recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto        M = make_kspace_pre(pre.Get(), recon->oshape, traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());
  auto const  sz = recon->ishape;

  Cropper     out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3         outSz = out_cropper.size();
  Cx5         allData = reader.readTensor<Cx5>();
  float const scale = Scaling(coreOpts.scaling, recon, M->cadjoint(CChipMap(allData, 0)));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5         out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();

  LSQR<ReconOp>      lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get()};
  LAD<LSQR<ReconOp>> lad{lsmr, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
  for (Index iv = 0; iv < volumes; iv++) {
    out.chip<4>(iv) = out_cropper.crop4(lad.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
  }

  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
