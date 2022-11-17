#include "types.hpp"

#include "basis.hpp"
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
  MultiFLAIR,
  MultiFLAIRInv,
  DWI
};

std::unordered_map<std::string, Sequences> SequenceMap{
  {"T1T2Prep", Sequences::T1T2},
  {"MPRAGE", Sequences::MPRAGE},
  {"DIR", Sequences::DIR},
  {"T2Prep", Sequences::T2Prep},
  {"T2InvPrep", Sequences::T2InvPrep},
  {"T2FLAIR", Sequences::T2FLAIR},
  {"MultiFLAIR", Sequences::MultiFLAIR},
  {"MultiFLAIRInv", Sequences::MultiFLAIRInv},
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
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index> spoil(parser, "N", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float> TI(parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.f);
  args::ValueFlag<float> Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float> te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);
  args::ValueFlag<float> bval(parser, "b", "b value", {'b', "bval"}, 0.f);

  args::ValueFlag<Index> nsamp(parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 2048);
  args::ValueFlag<float> thresh(parser, "T", "Threshold for SVD retention (default 95%)", {"thresh"}, 99.f);
  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 0);
  args::Flag demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag varimax(parser, "V", "Apply varimax rotation", {"varimax"});

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
    .bval = bval.Get(),
    .inversion = false};

  Eigen::ArrayXXf parameters, dynamics;
  switch (seq.Get()) {
  case Sequences::MPRAGE:
    std::tie(parameters, dynamics) = Simulate<rl::MPRAGE>(settings, nsamp.Get());
    break;
  case Sequences::DIR:
    std::tie(parameters, dynamics) = Simulate<rl::DIR>(settings, nsamp.Get());
    break;
  case Sequences::T2FLAIR:
    std::tie(parameters, dynamics) = Simulate<rl::T2FLAIR>(settings, nsamp.Get());
    break;
  case Sequences::MultiFLAIR:
    std::tie(parameters, dynamics) = Simulate<rl::MultiFLAIR>(settings, nsamp.Get());
    break;
  case Sequences::MultiFLAIRInv:
    settings.inversion = true;
    std::tie(parameters, dynamics) = Simulate<rl::MultiFLAIR>(settings, nsamp.Get());
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

  Basis basis(parameters, dynamics, thresh.Get(), nBasis.Get(), demean.Get(), varimax.Get());
  HD5::Writer writer(oname.Get());
  basis.write(writer);

  return EXIT_SUCCESS;
}
