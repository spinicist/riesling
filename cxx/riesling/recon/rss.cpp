#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"

using namespace rl;

void main_recon_rss(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nV = basis.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const A = Recon::Channels(coreOpts.ndft, gridOpts, traj, nC, nS, basis);
  auto const M = make_kspace_pre(traj, nC, basis, gridOpts.vcc, preOpts.type.Get(), preOpts.bias.Get());
  LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

  TOps::Crop<Cx, 5> outFOV(A->ishape, AddFront(traj.matrixForFOV(coreOpts.fov.Get()), nC, nV));
  Cx5               out(AddBack(LastN<4>(outFOV.ishape), nV));
  for (Index it = 0; it < nT; it++) {
    auto const channels = lsmr.run(&noncart(0, 0, 0, 0, it));
    auto const cropped = outFOV.forward(Tensorfy(channels, A->ishape));
    out.chip<4>(it) = (cropped * cropped.conjugate()).sum(Sz1{0}).sqrt();
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  Log::Print("Finished {}", parser.GetCommand().Name());
}
