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
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_lsmr(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto A = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto M = make_kspace_pre(pre.Get(), A->oshape, traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());
  auto debug = [&A](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), A->ishape, x.data());
  };
  LSMR lsmr{A, M, its.Get(), atol.Get(), btol.Get(), ctol.Get(), debug};
  auto sz = A->ishape;
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 const outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, A, M->adjoint(CChipMap(allData, 0)));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes), resid;
  if (coreOpts.residImage) {
    resid.resize(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  }

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto x = lsmr.run(&allData(0, 0, 0, 0, iv), λ.Get());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
    if (coreOpts.residImage || coreOpts.residKSpace) {
      allData.chip<4>(iv) -= A->forward(xm);
    }
    if (coreOpts.residImage) {
      xm = A->adjoint(allData.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm) / resid.chip<4>(iv).constant(scale);
    }
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved(), resid, allData);
  return EXIT_SUCCESS;
}
