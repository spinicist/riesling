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
  CoreOpts    coreOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  auto const  sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, coreOpts, traj, Cx5()), basis.dimension(0));
  auto const  recon = make_recon(coreOpts, sdcOpts, traj, sense, basis);
  auto const  shape = recon->ishape;
  auto        P = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  PDHG pdhg(A, P, reg);

  fmt::print("{:4.3E}\n{:4.3E}", fmt::join(pdhg.σ, ","), pdhg.τ);

  return EXIT_SUCCESS;
}
