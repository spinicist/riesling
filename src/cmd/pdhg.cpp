#include "types.hpp"

#include "algo/eig.hpp"
#include "algo/pdhg.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "prox/entropy.hpp"
#include "prox/llr.hpp"
#include "prox/thresh-wavelets.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_pdhg(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index> its(parser, "ITS", "Max iterations (4)", {"max-its"}, 4);

  args::ValueFlag<float> tv(parser, "TV", "Total Variation", {"tv"});
  args::ValueFlag<float> tvt(parser, "TVT", "Total Variation along time/frames/basis", {"tvt"});
  args::ValueFlag<float> l1(parser, "L1", "Simple L1 regularization", {"l1"});
  args::ValueFlag<float> nmrent(parser, "E", "NMR Entropy", {"nmrent"});

  args::ValueFlag<float> llr(parser, "L", "LLR regularization", {"llr"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);

  args::ValueFlag<Index> wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wavelet-levels"}, 4);
  args::ValueFlag<Index> waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6);

  args::ValueFlag<float> σin(parser, "σ", "Pre-computed σ", {"sigma"});
  args::ValueFlag<float> τin(parser, "τ", "Pre-computed τ", {"tau"}, -1.f);
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto A = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto const sz = A->ishape;
  auto P = make_kspace_pre(pre.Get(), A->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  std::shared_ptr<Prox::Prox<Cx>> prox;
  std::shared_ptr<Ops::Op<Cx>> G;

  if (wavelets) {
    G = std::make_shared<TensorIdentity<Cx, 4>>(sz);
    prox = std::make_shared<Prox::ThresholdWavelets>(wavelets.Get(), sz, waveWidth.Get(), waveLevels.Get());
  } else if (llr) {
    G = std::make_shared<TensorIdentity<Cx, 4>>(sz);
    prox = std::make_shared<Prox::LLR>(llr.Get(), llrPatch.Get(), llrWin.Get(), sz);
  } else if (nmrent) {
    G = std::make_shared<TensorIdentity<Cx, 4>>(sz);
    prox = std::make_shared<Prox::NMREntropy>(nmrent.Get(), G->rows());
  } else if (l1) {
    G = std::make_shared<TensorIdentity<Cx, 4>>(sz);
    prox = std::make_shared<Prox::SoftThreshold>(l1.Get(), G->rows());
  } else if (tv) {
    G = std::make_shared<GradOp>(sz, std::vector<Index>{1, 2, 3});
    prox = std::make_shared<Prox::SoftThreshold>(tv.Get(), G->rows());
  } else if (tvt) {
    G = std::make_shared<GradOp>(sz, std::vector<Index>{0});
    prox = std::make_shared<Prox::SoftThreshold>(tvt.Get(), G->rows());
  } else {
    Log::Fail("At least one regularizer must be specified");
  }

  float σ;
  if (σin) {
    σ = σin.Get();
  } else {
    auto eigG = PowerMethod(G, 32);
    σ = 1.f / eigG.val;
    Log::Print("σ {}", σ);
  }

  std::function<void(Index const, PDHG::Vector const &, PDHG::Vector const &, PDHG::Vector const &)> debug_x =
    [sz](Index const ii, PDHG::Vector const &x, PDHG::Vector const &x̅, PDHG::Vector const &xdiff) {
      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), sz, x.data());
      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), sz, x̅.data());
      Log::Tensor(fmt::format("pdhg-xdiff-{:02d}", ii), sz, xdiff.data());
    };

  PDHG pdhg{A, P, G, prox, its.Get(), debug_x};
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, A, P, &allData(0, 0, 0, 0, 0));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto x = pdhg.run(&allData(0, 0, 0, 0, iv), σ, τin.Get());
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved());
  return EXIT_SUCCESS;
}
