#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "precond.hpp"

using namespace rl;

int main_psf(args::Subparser &parser)
{
  CoreOpts                          coreOpts(parser);
  args::ValueFlag<std::string>      pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<Sz3, SzReader<3>> matrix(parser, "M", "Output matrix size", {"matrix", 'm'});
  args::ValueFlag<float>            preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index>            its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<float>            atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float>            btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float>            ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = 1;
  auto const  shape = matrix ? matrix.Get() : traj.matrixForFOV(coreOpts.fov.Get());

  std::shared_ptr<TensorOperator<Cx, 5, 4>> A = nullptr;
  if (coreOpts.ndft) {
    A = make_ndft(traj.points(), nC, shape, basis);
  } else {
    A = make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC, shape, basis, nullptr, coreOpts.bucketSize.Get(),
                   coreOpts.splitSize.Get());
  }
  auto const M = make_kspace_pre(pre.Get(), nC, traj, basis, preBias.Get(), coreOpts.ndft.Get());

  LSMR lsmr{A, M, its.Get(), atol.Get(), btol.Get(), ctol.Get()};
  Cx3  ones(1, traj.nSamples(), traj.nTraces());
  ones.setConstant(1.f / std::sqrt(Product(shape)));
  auto        x = lsmr.run(ones.data());
  auto        xm = Tensorfy(x, A->ishape);
  auto const  fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), "h5");
  HD5::Writer writer(fname);
  writer.writeTensor("psf", xm.dimensions(), xm.data());

  return EXIT_SUCCESS;
}
