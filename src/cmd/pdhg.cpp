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
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index> its(parser, "ITS", "Max iterations (4)", {"max-its"}, 4);

  args::ValueFlag<std::vector<float>, VectorReader<float>> σin(parser, "σ", "Pre-computed σ values", {"sigma"});
  args::ValueFlag<float> τin(parser, "τ", "Pre-computed τ", {"tau"}, -1.f);
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto const shape = recon->ishape;
  auto P = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers reg(regOpts, shape, A);

  std::function<void(Index const, PDHG::Vector const &, PDHG::Vector const &, PDHG::Vector const &)> debug_x =
    [shape](Index const ii, PDHG::Vector const &x, PDHG::Vector const &x̅, PDHG::Vector const &xdiff) {
      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, x.data());
      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data());
      Log::Tensor(fmt::format("pdhg-xdiff-{:02d}", ii), shape, xdiff.data());
    };

  PDHG pdhg{A, P, reg.ops, reg.prox, reg.σ(σin.Get()), its.Get(), debug_x};
  Cropper out_cropper(info.matrix, LastN<3>(shape), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, recon, P, &allData(0, 0, 0, 0, 0));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(shape[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto x = pdhg.run(&allData(0, 0, 0, 0, iv), τin.Get());
    auto xm = Tensorfy(x, shape);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved());
  return EXIT_SUCCESS;
}
