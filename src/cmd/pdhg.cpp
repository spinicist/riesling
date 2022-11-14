#include "types.hpp"

#include "algo/pdhg.hpp"
#include "cropper.h"
#include "func/dict.hpp"
#include "func/llr.hpp"
#include "func/thresh-wavelets.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "sdc.hpp"
#include "sense.hpp"

using namespace rl;

int main_pdhg(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<Index> its(parser, "ITS", "Max iterations (4)", {"max-its"}, 4);
  args::ValueFlag<float> τ(parser, "σ", "Dual step-size", {"tau"}, 1.f);

  args::ValueFlag<float> λ(parser, "λ", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> winSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);
  args::Flag wavelets(parser, "W", "Wavelets", {"wavelets", 'w'});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wave-levels"}, 4);
  args::ValueFlag<Index> waveSize(parser, "W", "Wavelet size (4/6/8)", {"wave-size"}, 6);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, false, reader);
  auto const sz = recon->inputDimensions();
  
  std::shared_ptr<Prox<Cx4>> reg;
  if (wavelets) {
    reg = std::make_shared<ThresholdWavelets>(sz, λ.Get(), waveSize.Get(), waveLevels.Get());
  } else {
    reg = std::make_shared<LLR>(λ.Get(), patchSize.Get(), winSize.Get());
  };

  auto sc = KSpaceSingle(traj);
  auto const odims = recon->outputDimensions();
  Cx4 P = sc.reshape(Sz4{1, odims[1], odims[2], 1}).broadcast(Sz4{odims[0], 1, 1, odims[3]}).cast<Cx>();
  PrimalDualHybridGradient<ReconOp> pdhg{recon, P, reg, its.Get()};

  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    out.chip<4>(iv) = out_cropper.crop4(pdhg.run(CChipMap(allData, iv), τ.Get()));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
