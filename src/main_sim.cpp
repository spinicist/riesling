#include "types.h"

#include "io.h"
#include "log.h"
#include "parse_args.h"
#include "sim-eddy.h"
#include "sim-mupa.h"
#include "sim-prep.h"

int main_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<long> sps(
    parser, "SEGMENT LENGTH", "Number of spokes/readouts per segment", {'s', "spokes"}, 128);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.01f);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.012f);
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
  args::ValueFlag<float> gLo(
    parser, "ɣ", "Low value for eddy-current angles (default -π)", {"eddylo"}, -M_PI);
  args::ValueFlag<float> gHi(
    parser, "ɣ", "High value for eddy-current angles (default π)", {"eddyhi"}, M_PI);

  args::Flag mupa(parser, "M", "Run a MUPA simulation", {"mupa"});

  args::ValueFlag<long> randomSamp(
    parser, "N", "Use N random parameter samples for dictionary", {"random"}, 0);
  args::ValueFlag<long> subsamp(
    parser, "S", "Subsample dictionary for SVD step (saves time)", {"subsamp"}, 1);
  args::ValueFlag<float> thresh(
    parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<long> nBasis(
    parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);

  Log log = ParseCommand(parser);

  Sim::Sequence const seq{
    .sps = sps.Get(),
    .alpha = alpha.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .TI = TI.Get(),
    .Trec = Trec.Get()};
  Sim::Parameter const T1{nT1.Get(), T1Lo.Get(), T1Hi.Get(), true};
  Sim::Parameter const beta{nb.Get(), bLo.Get(), bHi.Get(), bLog};
  Sim::Parameter const B1{nB1.Get(), B1Lo.Get(), B1Hi.Get(), false};
  Sim::Result result;
  if (ng) {
    Sim::Parameter const gamma{ng.Get(), gLo.Get(), gHi.Get(), false};
    result = Sim::Eddy(T1, beta, gamma, B1, seq, randomSamp.Get(), log);
  } else if (mupa) {
    Sim::Parameter const T2{65, 0.02, 0.2, true};
    result = Sim::MUPA(T1, T2, B1, seq, randomSamp.Get(), log);
  } else {
    result = Sim::Simple(T1, beta, seq, randomSamp.Get(), log);
  }

  // Normalize dictionary
  result.dynamics.rowwise().normalize();
  // Calculate SVD - observations are in rows
  log.info("Calculating SVD {}x{}", result.dynamics.cols() / subsamp.Get(), result.dynamics.rows());
  auto const svd = subsamp
                     ? result.dynamics(Eigen::seq(0, Eigen::last, subsamp.Get()), Eigen::all)
                         .matrix()
                         .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                     : result.dynamics.matrix().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
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
  log.info(
    "Retaining {} basis vectors, cumulative energy: {}", nRetain, cumsum.head(nRetain).transpose());
  // Scale and flip the basis vectors to always have a positive first element for stability
  Eigen::ArrayXf flip = Eigen::ArrayXf::Ones(nRetain);
  flip = (svd.matrixV().leftCols(nRetain).row(0).transpose().array() < 0.f).select(-flip, flip);
  Eigen::MatrixXf const basis =
    svd.matrixV().leftCols(nRetain).array().rowwise() * flip.transpose();
  log.info("Computing dictionary");
  Eigen::ArrayXXf Dk = result.dynamics.matrix() * basis;

  HD5::Writer writer(oname.Get(), log);
  writer.writeMatrix(basis, "basis");
  writer.writeMatrix(Dk, "dictionary");
  writer.writeMatrix(result.parameters, "parameters");
  writer.writeMatrix(result.Mz_ss, "Mz_ss");
  writer.writeMatrix(result.dynamics, "dynamics");
  return EXIT_SUCCESS;
}
