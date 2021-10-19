#include "types.h"

#include "io_hd5.h"
#include "log.h"
#include "parse_args.h"
#include "sim-eddy.h"
#include "sim-prep.h"

int main_basis_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<long> sps(
      parser, "SEGMENT LENGTH", "Number of spokes/readouts per segment", {'s', "spokes"}, 128);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> TI(
      parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.45f);
  args::ValueFlag<float> Trec(
      parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float> T1Lo(
      parser, "T1", "Low value of T1 for basis (default 0.4s)", {"T1lo"}, 0.4f);
  args::ValueFlag<float> T1Hi(
      parser, "T1", "High value of T1 for basis (default 5s)", {"T1hi"}, 5.f);
  args::ValueFlag<long> nT1(
      parser, "N", "Number of T1 values for basis (default 128)", {"nT1"}, 32);
  args::ValueFlag<float> bLo(parser, "β", "Low value for β (default -1)", {"betalo"}, -1.f);
  args::ValueFlag<float> bHi(parser, "β", "High value for β (default 1)", {"betahi"}, 1.f);
  args::ValueFlag<long> nb(
      parser, "N", "Number of beta values for basis (default 128)", {"nbeta"}, 32);
  args::Flag bLog(parser, "", "Use logarithmic spacing for β", {"betalog"});
  args::ValueFlag<long> ng(parser, "N", "Number of eddy-current angles", {"eddy"}, 32);
  args::Flag pc(parser, "PC", "Uses 0/180 phase-cycling", {"pc"});
  args::ValueFlag<float> thresh(
      parser, "T", "Threshold for SVD retention (default 0.05)", {"thresh"}, 0.05);
  args::ValueFlag<long> nBasis(
      parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);

  Log log = ParseCommand(parser);

  Sequence const seq{sps.Get(), alpha.Get(), TR.Get(), TI.Get(), Trec.Get()};
  SimResult results;
  if (ng) {
    results = Sim::Diffusion(
        nT1.Get(), T1Lo.Get(), T1Hi.Get(), nb.Get(), bLo.Get(), bHi.Get(), ng.Get(), seq, log);
  } else if (pc) {
    results = Sim::PhaseCycled(
        nT1.Get(), T1Lo.Get(), T1Hi.Get(), nb.Get(), bLo.Get(), bHi.Get(), seq, log);
  } else {
    results = Sim::Simple(
        nT1.Get(), T1Lo.Get(), T1Hi.Get(), nb.Get(), bLo.Get(), bHi.Get(), bLog, seq, log);
  }
  // Calculate SVD
  log.info("Calculating SVD {}x{}", results.dynamics.rows(), results.dynamics.cols());
  auto const svd = results.dynamics.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  float const flip = (svd.matrixV().leftCols(1)(0) < 0) ? -1.f : 1.f;

  long nRetain = 0;
  if (nBasis) {
    nRetain = nBasis.Get();
    log.info("Retaining {} basis vectors", nRetain);
  } else {
    Eigen::ArrayXf vals = svd.singularValues();
    float const valsSum = vals.sum();
    nRetain = (vals > (valsSum * thresh.Get())).count();
    log.info(
        "{} singular values are above {} threshold: {}",
        nRetain,
        thresh.Get(),
        vals.head(nRetain).transpose());
  }

  Eigen::MatrixXf const basisMat = flip * svd.matrixV().leftCols(nRetain);

  log.info("Computing dictionary");
  Eigen::MatrixXf const D = svd.matrixU().leftCols(nRetain) *
                            svd.singularValues().head(nRetain).asDiagonal() *
                            svd.matrixV().leftCols(nRetain).adjoint();
  Eigen::ArrayXXf Dk = D * basisMat;
  Dk = Dk.colwise() / Dk.rowwise().norm();

  Eigen::TensorMap<R2 const> dynamics(
      results.dynamics.data(), results.dynamics.rows(), results.dynamics.cols());
  Eigen::TensorMap<R2 const> basis(basisMat.data(), basisMat.rows(), basisMat.cols());
  Eigen::TensorMap<R2 const> dictionary(Dk.data(), Dk.rows(), Dk.cols());
  Eigen::TensorMap<R2 const> parameters(
      results.parameters.data(), results.parameters.rows(), results.parameters.cols());

  HD5::Writer writer(oname.Get(), log);
  writer.writeBasis(R2(basis));
  writer.writeDynamics(R2(dynamics));
  writer.writeRealMatrix(R2(dictionary), "dictionary");
  writer.writeRealMatrix(R2(parameters), "parameters");
  return EXIT_SUCCESS;
}
