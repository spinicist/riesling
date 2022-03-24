#include "types.h"

#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "sim/eddy.h"
#include "sim/flair.h"
#include "sim/mupa.h"
#include "sim/prep.h"

int main_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> sps(parser, "SPS", "Spokes per segment", {'s', "spokes"}, 128);
  args::ValueFlag<Index> gps(parser, "GPS", "Groups per segment", {'g', "gps"}, 1);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.01f);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.012f);
  args::ValueFlag<float> TI(
    parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.45f);
  args::ValueFlag<float> Trec(
    parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float> te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);
  args::ValueFlag<Index> nT1(
    parser, "N", "Number of T1 values for basis (default 512)", {"nT1"}, 512);
  args::ValueFlag<float> T1Lo(
    parser, "T1", "Low value of T1 for basis (default 0.4s)", {"T1lo"}, 0.4f);
  args::ValueFlag<float> T1Hi(
    parser, "T1", "High value of T1 for basis (default 5s)", {"T1hi"}, 5.f);
  args::ValueFlag<Index> nb(
    parser, "N", "Number of beta values for basis (default 128)", {"nbeta"}, 32);
  args::ValueFlag<float> bLo(parser, "β", "Low value for β (default -1)", {"betalo"}, -1.f);
  args::ValueFlag<float> bHi(parser, "β", "High value for β (default 1)", {"betahi"}, 1.f);
  args::Flag bLog(parser, "", "Use logarithmic spacing for β", {"betalog"});

  args::ValueFlag<Index> nB1(parser, "N", "Number of B1 values for basis (default 7)", {"nB1"}, 7);
  args::ValueFlag<float> B1Lo(parser, "B1", "Low value for B1 (default 0.7)", {"B1lo"}, 0.7f);
  args::ValueFlag<float> B1Hi(parser, "B1", "High value for B1 (default 1.3)", {"B1hi"}, 1.3f);

  args::ValueFlag<Index> ng(parser, "N", "Number of eddy-current angles", {"eddy"}, 32);
  args::ValueFlag<float> gLo(
    parser, "ɣ", "Low value for eddy-current angles (default -π)", {"eddylo"}, -M_PI);
  args::ValueFlag<float> gHi(
    parser, "ɣ", "High value for eddy-current angles (default π)", {"eddyhi"}, M_PI);

  args::ValueFlag<Index> randomSamp(
    parser, "N", "Use N random parameter samples for dictionary", {"random"}, 0);
  args::ValueFlag<Index> subsamp(
    parser, "S", "Subsample dictionary for SVD step (saves time)", {"subsamp"}, 1);
  args::ValueFlag<float> thresh(
    parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<Index> nBasis(
    parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);

  args::Flag mupa(parser, "M", "Run a MUPA simulation", {"mupa"});
  args::Flag flair(parser, "F", "Run a FLAIR simulation", {"flair"});
  args::Flag t1sim(parser, "T1", "Run a T1 prep simulation", {"T1sim"});
  args::Flag t2sim(parser, "T2", "Run a T1 prep simulation", {"T2sim"});
  args::Flag t1t2sim(parser, "T1T2", "Run a T1 prep simulation", {"T1T2sim"});

  ParseCommand(parser);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  Sim::Sequence const seq{
    .sps = sps.Get(),
    .gps = gps.Get(),
    .alpha = alpha.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .TI = TI.Get(),
    .Trec = Trec.Get(),
    .TE = te.Get()};
  fmt::print(FMT_STRING("nT1 {} \n"), nT1.Get());
  Sim::Parameter const T1{nT1.Get(), T1Lo.Get(), T1Hi.Get(), true};
  Sim::Parameter const beta{nb.Get(), bLo.Get(), bHi.Get(), bLog};
  Sim::Parameter const B1{nB1.Get(), B1Lo.Get(), B1Hi.Get(), false};
  Sim::Result result;
  if (ng) {
    Sim::Parameter const gamma{ng.Get(), gLo.Get(), gHi.Get(), false};
    result = Sim::Eddy(T1, beta, gamma, B1, seq, randomSamp.Get());
  } else if (mupa) {
    Sim::Parameter const T2{65, 0.02, 0.2, true};
    result = Sim::MUPA(T1, T2, B1, seq, randomSamp.Get());
  } else if (t1sim) {
    result = Sim::T1Prep(T1, seq, randomSamp.Get());
  } else if (t2sim) {
    Sim::Parameter const T2{65, 0.02, 0.2, true};
    result = Sim::T2Prep(T1, T2, seq, randomSamp.Get());
  } else if (t1t2sim) {
    Sim::Parameter const T2{65, 0.02, 0.2, true};
    result = Sim::T1T2Prep(T1, T2, seq, randomSamp.Get());
  } else if (flair) {
    Sim::Parameter const T2{65, 0.02, 0.2, true};
    result = Sim::FLAIR(T1, T2, seq, randomSamp.Get());
  } else {
    result = Sim::Simple(T1, beta, seq, randomSamp.Get());
  }

  // Calculate SVD - observations are in rows
  Log::Print(
    "Calculating SVD {}x{}", result.dynamics.cols() / subsamp.Get(), result.dynamics.rows());
  auto const svd = subsamp
                     ? result.dynamics(Eigen::seq(0, Eigen::last, subsamp.Get()), Eigen::all)
                         .matrix()
                         .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                     : result.dynamics.matrix().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  float const nullThresh = svd.singularValues()[0] * std::numeric_limits<float>::epsilon();
  Index const nullCount = (svd.singularValues().array() > nullThresh).count();
  fmt::print(FMT_STRING("{} values above null-space threshold {}\n"), nullCount, nullThresh);
  Eigen::ArrayXf const vals = svd.singularValues().array().square();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum = 100.f * cumsum / cumsum.tail(1)[0];
  Index nRetain = 0;
  if (nBasis) {
    nRetain = nBasis.Get();
  } else {
    nRetain = (cumsum < thresh.Get()).count();
  }
  Log::Print(
    "Retaining {} basis vectors, cumulative energy: {}", nRetain, cumsum.head(nRetain).transpose());
  // Scale and flip the basis vectors to always have a positive first element for stability
  Eigen::ArrayXf flip = Eigen::ArrayXf::Ones(nRetain);
  flip = (svd.matrixV().leftCols(nRetain).row(0).transpose().array() < 0.f).select(-flip, flip);
  Eigen::MatrixXf const basis =
    svd.matrixV().leftCols(nRetain).array().rowwise() * flip.transpose();
  Log::Print("Computing dictionary");
  Eigen::ArrayXXf dict = result.dynamics.matrix() * basis;
  Eigen::ArrayXf const norm = dict.rowwise().norm();
  dict.rowwise().normalize();

  HD5::Writer writer(oname.Get());
  writer.writeMatrix(basis, HD5::Keys::Basis);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(result.parameters, HD5::Keys::Parameters);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(result.dynamics, HD5::Keys::Dynamics);
  return EXIT_SUCCESS;
}
