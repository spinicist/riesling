#include "types.h"

#include "algo/admm.hpp"
#include "algo/llr.h"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "precond/single.hpp"
#include "sdc.h"
#include "sense.h"

int main_admm(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});

  args::Flag use_cg(parser, "C", "Use CG instead of LSQR for inner loop", {"cg"});
  args::Flag use_lsmr(parser, "L", "Use LSMR instead of LSQR for inner loop", {"lsmr"});

  args::Flag precond(parser, "P", "Apply Ong's single-channel M-conditioner", {"pre"});
  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (2)", {"max-its"}, 2);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "Max outer iterations (8)", {"max-outer-its"}, 8);
  args::ValueFlag<float> abstol(parser, "ABS", "Outer absolute tolerance (1e-3)", {"abs-tol"}, 1.e-3f);
  args::ValueFlag<float> reltol(parser, "REL", "Outer relative tolerance (1e-3)", {"rel-tol"}, 1.e-3f);
  args::ValueFlag<float> ρ(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 0.1f);

  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.fast);
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);

  std::unique_ptr<Precond<Cx3>> M = precond ? std::make_unique<SingleChannel>(traj, kernel.get()) : nullptr;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), info.channels, basis, core.fast);
  }
  ReconOp recon(gridder.get(), senseMaps, sdc.get());

  auto reg = [&](Cx4 const &x) -> Cx4 { return llr_sliding(x, λ.Get() / ρ.Get(), patchSize.Get()); };

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    if (use_cg) {
      vol = admm_cg(
        outer_its.Get(),
        inner_its.Get(),
        atol.Get(),
        recon,
        reg,
        ρ.Get(),
        reader.noncartesian(iv),
        abstol.Get(),
        reltol.Get());
    } else if (use_lsmr) {
      vol = admm_lsmr(
        outer_its.Get(),
        ρ.Get(),
        reg,
        inner_its.Get(),
        recon,
        reader.noncartesian(iv),
        M.get(),
        atol.Get(),
        btol.Get(),
        ctol.Get(),
        abstol.Get(),
        reltol.Get());
    } else {
      vol = admm_lsqr(
        outer_its.Get(),
        ρ.Get(),
        reg,
        inner_its.Get(),
        recon,
        reader.noncartesian(iv),
        M.get(),
        atol.Get(),
        btol.Get(),
        ctol.Get(),
        abstol.Get(),
        reltol.Get());
    }
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "admm", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
