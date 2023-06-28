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

int main_pdhg_setup(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto const shape = recon->ishape;
  auto P = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()));

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers reg(regOpts, shape, A);

  PDHG pdhg(A, P, reg);

  fmt::print("{:5.3E}\n{:5.3E}", fmt::join(pdhg.σ, ","), pdhg.τ);

  return EXIT_SUCCESS;
}
