#include "types.hpp"

#include "algo/eig.hpp"
#include "algo/pdhg.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_pdhg(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  args::ValueFlag<std::string> scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu");
  args::ValueFlag<Index>       its(parser, "ITS", "Max iterations (4)", {"max-its"}, 4);
  args::ValueFlag<std::vector<float>, VectorReader<float>> σin(parser, "σ", "Pre-computed dual step sizes", {"sigma"});
  args::ValueFlag<float>                                   τin(parser, "τ", "Pre-computed primal step size", {"tau"}, -1.f);
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nS = noncart.dimension(3);
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const recon = Recon::SENSE(coreOpts, gridOpts, senseOpts, traj, nS, basis, noncart);
  auto const shape = recon->ishape;
  auto const P = make_kspace_pre(traj, recon->oshape[0], ReadBasis(coreOpts.basisFile.Get()), gridOpts.vcc, preOpts.type.Get(),
                                 preOpts.bias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  std::function<void(Index const, PDHG::Vector const &, PDHG::Vector const &, PDHG::Vector const &)> debug_x =
    [shape](Index const ii, PDHG::Vector const &x, PDHG::Vector const &x̅, PDHG::Vector const &xdiff) {
      Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, x.data());
      Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data());
      Log::Tensor(fmt::format("pdhg-xdiff-{:02d}", ii), shape, xdiff.data());
    };

  PDHG        pdhg(A, P, reg.regs, σin.Get(), τin.Get(), debug_x);
  
  TOps::Crop<Cx, 4> oc(recon->ishape, AddFront(traj.matrixForFOV(coreOpts.fov.Get()), recon->ishape[0]));
  Cx5               out(AddBack(oc.oshape, nV));

  float const scale = Scaling(scaling, recon, P, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);

  for (Index iv = 0; iv < nV; iv++) {
    auto x = pdhg.run(&noncart(0, 0, 0, 0, iv), its.Get());
    auto xm = Tensorfy(x, recon->ishape);
    out.chip<4>(iv) = oc.forward(xm) / out.chip<4>(iv).constant(scale);
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  rl::Log::Print("Finished {}", parser.GetCommand().Name());
}
