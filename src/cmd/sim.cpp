#include "types.hpp"

#include "algo/decomp.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sim/dir.hpp"
#include "sim/dwi.hpp"
#include "sim/mprage.hpp"
#include "sim/parameter.hpp"
#include "sim/t1t2.hpp"
#include "sim/t2flair.hpp"
#include "sim/t2prep.hpp"
#include "threads.hpp"

using namespace rl;

template <typename T>
auto Simulate(rl::Settings const &s, Index const nsamp)
{
  T simulator{s};

  Eigen::ArrayXXf parameters = simulator.parameters(nsamp);
  Eigen::ArrayXXf dynamics(simulator.length(), parameters.cols());
  auto const start = Log::Now();
  auto task = [&](Index const ii) { dynamics.col(ii) = simulator.simulate(parameters.col(ii)); };
  Threads::For(task, parameters.cols(), "Simulation");
  Log::Print(FMT_STRING("Simulation took {}"), Log::ToNow(start));
  return std::make_tuple(parameters, dynamics);
}

enum struct Sequences
{
  T1T2 = 0,
  MPRAGE,
  DIR,
  T2Prep,
  T2InvPrep,
  T2FLAIR,
  DWI
};

std::unordered_map<std::string, Sequences> SequenceMap{
  {"T1T2Prep", Sequences::T1T2},
  {"MPRAGE", Sequences::MPRAGE},
  {"DIR", Sequences::DIR},
  {"T2Prep", Sequences::T2Prep},
  {"T2InvPrep", Sequences::T2InvPrep},
  {"T2FLAIR", Sequences::T2FLAIR},
  {"DWI", Sequences::DWI}};

int main_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::MapFlag<std::string, Sequences> seq(parser, "T", "Sequence type (default T1T2)", {"seq"}, SequenceMap);
  args::ValueFlag<Index> spg(parser, "SPG", "traces per group", {'s', "spg"}, 128);
  args::ValueFlag<Index> gps(parser, "GPS", "Groups per segment", {'g', "gps"}, 1);
  args::ValueFlag<Index> gprep2(parser, "G", "Groups before prep 2", {"gprep2"}, 0);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> ascale(parser, "A", "Flip-angle scaling", {"ascale"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.01f);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.012f);
  args::ValueFlag<float> TI(parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.45f);
  args::ValueFlag<float> Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float> te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);
  args::ValueFlag<float> bval(parser, "b", "b value", {'b', "bval"}, 0.f);

  args::ValueFlag<Index> nsamp(parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 2048);
  args::ValueFlag<Index> subsamp(parser, "S", "Subsample dictionary for SVD step (saves time)", {"subsamp"}, 1);
  args::ValueFlag<float> thresh(parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);
  args::Flag varimax(parser, "V", "Apply varimax rotation", {"varimax"});

  ParseCommand(parser);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  rl::Settings const settings{
    .spg = spg.Get(),
    .gps = gps.Get(),
    .gprep2 = gprep2.Get(),
    .alpha = alpha.Get(),
    .ascale = ascale.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .TI = TI.Get(),
    .Trec = Trec.Get(),
    .TE = te.Get(),
    .bval = bval.Get()};

  Eigen::ArrayXXf parameters, dynamics;
  switch (seq.Get()) {
  case Sequences::MPRAGE:
    std::tie(parameters, dynamics) = Simulate<rl::MPRAGE>(settings, nsamp.Get());
    break;
  case Sequences::DIR:
    std::tie(parameters, dynamics) = Simulate<rl::DIR>(settings, nsamp.Get());
  case Sequences::T2FLAIR:
    std::tie(parameters, dynamics) = Simulate<rl::T2FLAIR>(settings, nsamp.Get());
    break;
  case Sequences::T2Prep:
    std::tie(parameters, dynamics) = Simulate<rl::T2Prep>(settings, nsamp.Get());
    break;
  case Sequences::T2InvPrep:
    std::tie(parameters, dynamics) = Simulate<rl::T2InvPrep>(settings, nsamp.Get());
    break;
  case Sequences::T1T2:
    std::tie(parameters, dynamics) = Simulate<rl::T1T2Prep>(settings, nsamp.Get());
    break;
  case Sequences::DWI:
    std::tie(parameters, dynamics) = Simulate<rl::DWI>(settings, nsamp.Get());
    break;
  }

  // Calculate SVD - observations are in cols
  auto const svd =
    SVD<float>(subsamp ? dynamics(Eigen::seq(0, Eigen::last, subsamp.Get()), Eigen::all) : dynamics, true, true);
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
  Log::Print("Retaining {} basis vectors, cumulative energy: {}", nRetain, cumsum.head(nRetain).transpose());
  // Scale and flip the basis vectors to always have a positive first element for stability
  // Eigen::ArrayXf flip = Eigen::ArrayXf::Ones(nRetain);
  // flip = (svd.V.leftCols(nRetain).row(0).transpose().array() < 0.f).select(-flip, flip);
  Eigen::MatrixXf basis = svd.V.leftCols(nRetain).array();
  if (varimax) {
    Log::Print("SIM Applying varimax rotation");
    float gamma = 1.0f;
    float const tol = 1e-6f;
    float q = 20;
    Index const p = basis.rows();
    Index const k = basis.cols();
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(k, k);
    float d = 0.f;
    for (Index ii = 0; ii < q; ii++) {
      float const d_old = d;
      Eigen::MatrixXf const λ = basis * R;
      Eigen::MatrixXf const x = basis.transpose() * (λ.array().pow(3.f).matrix() -
                                                     (λ * (λ.transpose() * λ).diagonal().asDiagonal()) * (gamma / p));
      auto const svdv = SVD<float>(x);
      R = svdv.U * svdv.V.adjoint();
      d = svdv.vals.sum();
      if (d_old != 0.f && (d / d_old) < 1 + tol)
        break;
    }
    basis = basis * R;
  }

  Eigen::ArrayXf const scales = svd.vals.head(nRetain) / svd.vals(0);
  Log::Print("Computing dictionary");
  Eigen::MatrixXf dict = basis.transpose() * dynamics.matrix();
  Eigen::ArrayXf const norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();

  HD5::Writer writer(oname.Get());
  writer.writeMatrix(basis, HD5::Keys::Basis);
  writer.writeMatrix(scales, HD5::Keys::Scales);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(parameters, HD5::Keys::Parameters);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(dynamics, HD5::Keys::Dynamics);
  return EXIT_SUCCESS;
}
