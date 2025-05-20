#include "types.hpp"

#include "algo/eig.hpp"
#include "algo/pdhg.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "precon.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_pdhg_setup(args::Subparser &parser)
{
  CoreArgs<3> coreArgs(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  ParseCommand(parser, coreArgs.iname);

  HD5::Reader reader(coreArgs.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  auto const  nS = noncart.dimension(3);
  auto const  nT = noncart.dimension(4);
  auto const  basis = ReadBasis(coreArgs.basisFile.Get());
  auto const  recon = Recon::SENSE(coreArgs.ndft, gridOpts, senseOpts, traj, nS, nT, basis, noncart);
  auto const  shape = recon->ishape;
  auto const  P =
    make_kspace_pre(traj, recon->oshape[0], ReadBasis(coreArgs.basisFile.Get()), preOpts.type.Get(), preOpts.bias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  PDHG pdhg(A, P, reg.regs);

  fmt::print("{:4.3E}\n{:4.3E}", fmt::join(pdhg.σ, ","), pdhg.τ);
}
