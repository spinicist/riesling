#include "types.hpp"

#include "basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sim/dir.hpp"
#include "sim/dwi.hpp"
#include "sim/ir.hpp"
#include "sim/parameter.hpp"
#include "sim/t1t2.hpp"
#include "sim/t2flair.hpp"
#include "sim/t2prep.hpp"
#include "threads.hpp"

using namespace rl;

template <typename T>
auto Run(rl::Settings const &s, Index const nsamp, std::vector<std::vector<float>> los, std::vector<std::vector<float>> his)
{
  if (los.size() != his.size()) {
    Log::Fail("Different number of parameter low bounds and high bounds");
  }
  if (los.size() == 0) {
    Log::Fail("Must specify at least one set of tissue parameters");
  }

  T simulator{s};
  Index const nP = T::nParameters;
  Eigen::ArrayXXf parameters(nP, nsamp * los.size());
  parameters.setZero();
  Index totalP = 0;
  for (size_t ii = 0; ii < los.size(); ii++) {
    Log::Print("Parameter set {}/{}. Low {} High {}", ii+1, los.size(), fmt::join(los[ii], "/"), fmt::join(his[ii], "/"));
    auto p = simulator.parameters(nsamp, los[ii], his[ii]);
    parameters.middleCols(totalP, p.cols()) = p;
    totalP += p.cols();
  }
  parameters.conservativeResize(nP, totalP);
  Eigen::ArrayXXf dynamics(simulator.length(), totalP);
  auto const start = Log::Now();
  auto task = [&](Index const ii) { dynamics.col(ii) = simulator.simulate(parameters.col(ii)); };
  Threads::For(task, totalP, "Simulation");
  Log::Print("Simulation took {}", Log::ToNow(start));
  return std::make_tuple(parameters, dynamics);
}

enum struct Sequences
{
  T1T2 = 0,
  IR,
  IR2,
  DIR,
  DIR2,
  T2Prep,
  T2InvPrep,
  T2FLAIR,
  DWI
};

std::unordered_map<std::string, Sequences> SequenceMap{
  {"T1T2Prep", Sequences::T1T2},
  {"IR", Sequences::IR},
  {"IR2", Sequences::IR2},
  {"DIR", Sequences::DIR},
  {"DIR2", Sequences::DIR2},
  {"T2Prep", Sequences::T2Prep},
  {"T2InvPrep", Sequences::T2InvPrep},
  {"T2FLAIR", Sequences::T2FLAIR},
  {"DWI", Sequences::DWI}};

int main_basis_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::MapFlag<std::string, Sequences> seq(parser, "T", "Sequence type (default T1T2)", {"seq"}, SequenceMap);
  args::ValueFlag<Index> spg(parser, "SPG", "traces per group", {'s', "spg"}, 128);
  args::ValueFlag<Index> gps(parser, "GPS", "Groups per segment", {'g', "gps"}, 1);
  args::ValueFlag<Index> gprep2(parser, "G", "Groups before prep 2", {"gprep2"}, 0);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> ascale(parser, "A", "Flip-angle scaling", {"ascale"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index> spoil(parser, "N", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float> TI(parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.f);
  args::ValueFlag<float> Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float> te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);
  args::ValueFlag<float> Tsat(parser, "TSAT", "Fat sat time", {"tsat"}, 0.f);
  args::ValueFlag<float> bval(parser, "b", "b value", {'b', "bval"}, 0.f);

  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pLo(
    parser, "LO", "Low values for parameters", {"lo"});
  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pHi(
    parser, "HI", "High values for parameters", {"hi"});
  args::ValueFlag<Index> nsamp(parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 2048);
  args::ValueFlag<float> thresh(parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);
  args::Flag demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag rotate(parser, "V", "Rotate basis", {"rotate"});
  args::Flag normalize(parser, "N", "Normalize dynamics before SVD", {"normalize"});

  ParseCommand(parser);
  if (!oname) {
    throw args::Error("No output filename specified");
  }

  rl::Settings settings{
    .spg = spg.Get(),
    .gps = gps.Get(),
    .gprep2 = gprep2.Get(),
    .spoil = spoil.Get(),
    .alpha = alpha.Get(),
    .ascale = ascale.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .TI = TI.Get(),
    .Trec = Trec.Get(),
    .TE = te.Get(),
    .Tsat = Tsat.Get(),
    .bval = bval.Get(),
    .inversion = false};

  Eigen::ArrayXXf pars, dyns;
  switch (seq.Get()) {
  case Sequences::IR: std::tie(pars, dyns) = Run<rl::IR>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::IR2: std::tie(pars, dyns) = Run<rl::IR2>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::DIR: std::tie(pars, dyns) = Run<rl::DIR>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::DIR2: std::tie(pars, dyns) = Run<rl::DIR2>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::T2FLAIR: std::tie(pars, dyns) = Run<rl::T2FLAIR>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::T2Prep: std::tie(pars, dyns) = Run<rl::T2Prep>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::T2InvPrep: std::tie(pars, dyns) = Run<rl::T2InvPrep>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::T1T2: std::tie(pars, dyns) = Run<rl::T1T2Prep>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  case Sequences::DWI: std::tie(pars, dyns) = Run<rl::DWI>(settings, nsamp.Get(), pLo.Get(), pHi.Get()); break;
  }

  Basis basis(dyns, thresh.Get(), nBasis.Get(), demean, rotate, normalize);
  HD5::Writer writer(oname.Get());
  basis.write(writer);
  writer.writeMatrix(pars, HD5::Keys::Parameters);
  return EXIT_SUCCESS;
}
