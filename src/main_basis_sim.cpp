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
  args::ValueFlag<long> nT1(
      parser, "N", "Number of T1 values for basis (default 128)", {"nT1"}, 32);
  args::ValueFlag<float> T1Lo(
      parser, "T1", "Low value of T1 for basis (default 0.4s)", {"T1lo"}, 0.4f);
  args::ValueFlag<float> T1Hi(
      parser, "T1", "High value of T1 for basis (default 5s)", {"T1hi"}, 5.f);
  args::ValueFlag<long> nb(
      parser, "N", "Number of beta values for basis (default 128)", {"nbeta"}, 32);
  args::ValueFlag<float> bLo(parser, "β", "Low value for β (default -1)", {"betalo"}, -1.f);
  args::ValueFlag<float> bHi(parser, "β", "High value for β (default 1)", {"betahi"}, 1.f);
  args::Flag bLog(parser, "", "Use logarithmic spacing for β", {"betalog"});

  args::ValueFlag<long> nB1(parser, "N", "Number of B1 values for basis (default 7)", {"nB1"}, 7);
  args::ValueFlag<float> B1Lo(parser, "B1", "Low value for B1 (default 0.7)", {"B1lo"}, 0.7f);
  args::ValueFlag<float> B1Hi(parser, "B1", "High value for B1 (default 1.3)", {"B1hi"}, 1.3f);

  args::ValueFlag<long> ng(parser, "N", "Number of eddy-current angles", {"eddy"}, 32);
  args::ValueFlag<float> gLo(parser, "ɣ", "Low value for eddy-current angles (default -π)", {"eddylo"}, -M_PI);
  args::ValueFlag<float> gHi(parser, "ɣ", "High value for eddy-current angles (default π)", {"eddyhi"}, M_PI);

  args::ValueFlag<float> thresh(
      parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 95.f);
  args::ValueFlag<long> nBasis(
      parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);

  Log log = ParseCommand(parser);

  Sim::Sequence const seq{sps.Get(), alpha.Get(), TR.Get(), TI.Get(), Trec.Get()};
  Sim::Parameter const T1{nT1.Get(), T1Lo.Get(), T1Hi.Get(), true};
  Sim::Parameter const beta{nb.Get(), bLo.Get(), bHi.Get(), bLog};
  Sim::Parameter const B1{nB1.Get(), B1Lo.Get(), B1Hi.Get(), false};
  Sim::Result results;
  if (ng) {
    Sim::Parameter const gamma{ng.Get(), gLo.Get(), gHi.Get(), false};
    results = Sim::Eddy(T1, beta, gamma, B1,  seq, log);
  } else {
    results = Sim::Simple(T1, beta, B1, seq, log);
  }
  long const nT = results.dynamics.cols(); // Number of timepoints in sim

  // Calculate SVD
  log.info("Calculating SVD {}x{}", results.dynamics.rows(), results.dynamics.cols());
  auto const svd = results.dynamics.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::ArrayXf const vals = svd.singularValues().array().square();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum = 100.f * cumsum / cumsum.tail(1)[0];
  long nRetain = 0;
  if (nBasis) {
    nRetain = nBasis.Get();
  } else {
    nRetain = (cumsum < thresh.Get()).count();
  }
  log.info("Retaining {} basis vectors, cumulative energy: {}\n",
           nRetain,
           cumsum.head(nRetain).transpose());
  float const flip = (svd.matrixV().leftCols(1)(0) < 0) ? -1.f : 1.f;
  Eigen::MatrixXf const basisMat = flip * svd.matrixV().leftCols(nRetain) * std::sqrt(nT);
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
