#include "types.hpp"

#include "algo/lsmr.hpp"
#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "precond.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

int main_psf(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  GridOpts gridOpts(parser);

  args::Flag                        mtf(parser, "M", "Save Modulation Transfer Function", {"mtf"});
  args::ValueFlag<Sz3, SzReader<3>> matrix(parser, "M", "Output matrix size", {"matrix", 'm'});

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);

  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Eigen::Array2f, Array2fReader> phases(parser, "P", "Phase accrued at start and end of spoke",
                                                        {"phases", 'p'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = 1;
  Index const nB = basis.dimension(0);
  auto const  shape = matrix ? matrix.Get() : traj.matrixForFOV(coreOpts.fov.Get());

  std::shared_ptr<TensorOperator<Cx, 5, 4>> A = nullptr;
  if (coreOpts.ndft) {
    A = make_ndft(traj.points(), nC, shape, basis);
  } else {
    A = make_nufft(traj, gridOpts, nC, shape, basis, nullptr);
  }
  auto const M = make_kspace_pre(pre.Get(), nC, traj, basis, preBias.Get(), coreOpts.ndft.Get());
  LSMR       lsmr{A, M, its.Get(), atol.Get(), btol.Get(), ctol.Get()};

  float const startPhase = phases.Get()[0];
  float const endPhase = phases.Get()[1];

  Eigen::VectorXcf const trace =
    Eigen::ArrayXcf::LinSpaced(traj.nSamples(), startPhase * 1if, endPhase * 1if).exp() * std::sqrt(nB / (float)Product(shape));
  Eigen::TensorMap<Cx1 const> traceM(trace.data(), Sz1{traj.nSamples()});

  Cx3         ks = traceM.reshape(Sz3{1, traj.nSamples(), 1}).broadcast(Sz3{1, 1, traj.nTraces()});
  auto        x = lsmr.run(ks.data());
  auto        xm = Tensorfy(x, LastN<4>(A->ishape));
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor("psf", xm.dimensions(), xm.data());

  if (mtf) {
    auto const fft = FFT::Make<4, 3>(xm.dimensions());
    Log::Print("Calculating MTF");
    xm *= xm.constant(std::sqrt(Product(shape)));
    fft->forward(xm);
    writer.writeTensor("mtf", xm.dimensions(), xm.data());
  }
  Log::Print("Finished");
  return EXIT_SUCCESS;
}
