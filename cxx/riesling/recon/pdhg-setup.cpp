#include "types.hpp"

#include "algo/eig.hpp"
#include "algo/pdhg.hpp"
#include "cropper.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_pdhg_setup(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  auto const  sense = std::make_shared<TOps::SENSE>(SENSE::Choose(senseOpts, gridOpts, traj, Cx5()), basis.dimension(0));
  auto const  recon = Recon::SENSE(coreOpts, gridOpts, traj, reader.dimensions()[3], sense, basis);
  auto const  shape = recon->ishape;
  auto const  P = make_kspace_pre(traj, recon->oshape[0], ReadBasis(coreOpts.basisFile.Get()), gridOpts.vcc, preOpts.type.Get(),
                                  preOpts.bias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  PDHG pdhg(A, P, reg.regs);

  fmt::print("{:4.3E}\n{:4.3E}", fmt::join(pdhg.σ, ","), pdhg.τ);
}
