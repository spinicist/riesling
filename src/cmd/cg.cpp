#include "types.hpp"

#include "algo/cg.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "scaling.hpp"
#include "sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_cg(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "pipe");
  SENSE::Opts senseOpts(parser);
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> thr(parser, "T", "Termination threshold (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {"max-its"}, 8);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, toeplitz, reader);
  auto normEqs = std::make_shared<NormalOp<Cx>>(recon);
  ConjugateGradients cg{normEqs, its.Get(), thr.Get(), true};

  auto sz = recon->ishape;
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, recon, CChipMap(allData, 0));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const &vol_start = Log::Now();
    auto b = recon->adjoint(CChipMap(allData, iv));
    out.chip<4>(iv) = out_cropper.crop4(Tensorfy(cg.run(b.data()), sz)) / out.chip<4>(iv).constant(scale);
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
