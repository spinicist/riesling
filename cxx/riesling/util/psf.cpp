#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "precon.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

void main_psf(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  args::Flag                        mtf(parser, "M", "Save Modulation Transfer Function", {"mtf"});
  args::ValueFlag<Sz3, SzReader<3>> matrix(parser, "M", "Output matrix size", {"matrix", 'm'});

  args::ValueFlag<Eigen::Array2f, Array2fReader> phases(parser, "P", "Phase accrued at start and end of spoke",
                                                        {"phases", 'p'});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = 1;
  Index const nB = basis.dimension(0);
  auto const  shape = matrix ? matrix.Get() : traj.matrixForFOV(coreOpts.fov.Get());

  std::shared_ptr<TOps::TOp<Cx, 5, 3>> A = nullptr;
  if (coreOpts.ndft) {
    A = TOps::NDFT<3>::Make(shape, traj.points(), nC, basis);
  } else {
    A = TOps::NUFFT<3>::Make(traj, gridOpts, nC, basis, shape);
  }
  auto const M = MakeKspacePre(traj, nC, 1, basis, preOpts.type.Get(), preOpts.bias.Get(), coreOpts.ndft.Get());
  LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

  float const startPhase = phases.Get()[0];
  float const endPhase = phases.Get()[1];

  Eigen::VectorXcf const trace =
    Eigen::ArrayXcf::LinSpaced(traj.nSamples(), startPhase * 1if, endPhase * 1if).exp() * std::sqrt(nB / (float)Product(shape));
  Eigen::TensorMap<Cx1 const> traceM(trace.data(), Sz1{traj.nSamples()});

  Cx3         ks = traceM.reshape(Sz3{1, traj.nSamples(), 1}).broadcast(Sz3{1, 1, traj.nTraces()});
  auto        x = lsmr.run(ks.data());
  auto        xm = Tensorfy(x, LastN<4>(A->ishape));
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, xm.dimensions(), xm.data(), {"v", "x", "y", "z"});

  if (mtf) {
    auto const fft = TOps::FFT<4, 3>(xm.dimensions());
    Log::Print("Calculating MTF");
    xm *= xm.constant(std::sqrt(Product(shape)));
    fft.forward(xm);
    writer.writeTensor("mtf", xm.dimensions(), xm.data(), {"v", "x", "y", "z"});
  }
  Log::Print("Finished");
}
