#include "types.h"

#include "algo/decomp.h"
#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "sim/parameter.h"
#include "sim/t1t2.h"

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

  args::ValueFlag<Index> nsamp(
    parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 2048);
  args::ValueFlag<Index> subsamp(
    parser, "S", "Subsample dictionary for SVD step (saves time)", {"subsamp"}, 1);
  args::ValueFlag<float> thresh(
    parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<Index> nBasis(
    parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);

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

  // T1, T2, B1
  Sim::Tissue wm({{0.8, 0.25, 0.5, 2.0}, {0.05, 0.025, 0.01, 0.25}});
  Sim::Tissue gm({{1.2, 0.25, 0.5, 2.0}, {0.075, 0.025, 0.01, 0.25}});
  Sim::Tissue csf({{3.5, 0.5, 2.5, 4.5}, {1.0, 0.4, 0.5, 2.5}});

  Sim::Tissues tissues({wm, gm, csf});
  auto const parameters = tissues.values(nsamp.Get());

  Eigen::ArrayXXf dynamics(parameters.cols(), 2 * seq.sps);
  for (Index ii = 0; ii < parameters.cols(); ii++) {
    dynamics.row(ii) = T1T2Prep(seq, parameters(0, ii), parameters(1, ii), 1.f);
  }

  // Calculate SVD - observations are in rows
  Log::Print("Calculating SVD {}x{}", dynamics.cols() / subsamp.Get(), dynamics.rows());
  auto const svd =
    SVD(subsamp ? dynamics(Eigen::seq(0, Eigen::last, subsamp.Get()), Eigen::all) : dynamics);

  float const nullThresh = svd.vals[0] * std::numeric_limits<float>::epsilon();
  Index const nullCount = (svd.vals > nullThresh).count();
  fmt::print(FMT_STRING("{} values above null-space threshold {}\n"), nullCount, nullThresh);
  Eigen::ArrayXf const vals = svd.vals.square();
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
  flip = (svd.vecs.leftCols(nRetain).row(0).transpose().array() < 0.f).select(-flip, flip);
  Eigen::MatrixXf const basis = svd.vecs.leftCols(nRetain).array().rowwise() * flip.transpose();
  Eigen::ArrayXf const scales = 1.f / vals.head(nRetain);
  Log::Print("Computing dictionary");
  Eigen::ArrayXXf dict = dynamics.matrix() * basis;
  Eigen::ArrayXf const norm = dict.rowwise().norm();
  dict.rowwise().normalize();

  HD5::Writer writer(oname.Get());
  writer.writeMatrix(basis, HD5::Keys::Basis);
  writer.writeMatrix(scales, HD5::Keys::Scales);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(parameters, HD5::Keys::Parameters);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(dynamics, HD5::Keys::Dynamics);
  return EXIT_SUCCESS;
}
