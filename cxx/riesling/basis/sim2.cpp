#include "types.hpp"

#include "basis/svd.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sim/parameter.hpp"
#include "threads.hpp"

#include "unsupported/Eigen/MatrixFunctions"

#include <complex>
#include <numbers>

using namespace rl;
using namespace std::literals::complex_literals;

struct Settings
{
  Index samplesPerSpoke = 256, samplesInGap = 2, spokesPerSeg = 256, spokesSpoil = 0, segsPerPrep = 2, segsPrep2 = 1,
        segsKeep = 2;
  float tSamp = 10e-6, alpha = 1.f, ascale = 1.f, TR = 2.e-3f, Tramp = 10.e-3f, Tssi = 10.e-3f, Tprep = 0, Trec = 0;
};

auto T1β1β2ω(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  Parameters::CheckSizes(4, {0.6f, -1.f, -1.f, -1000.f}, {4.3f, 1.f, 1.f, 1000.f}, lo, hi);
  float const R1lo = 1.f / lo[0];
  float const R1hi = 1.f / hi[0];
  float const β1lo = lo[1];
  float const β1hi = hi[1];
  float const β2lo = lo[2];
  float const β2hi = hi[2];
  float const ωlo = lo[3] * 2.f * M_PI;
  float const ωhi = hi[3] * 2.f * M_PI;
  Index const nω = 10;
  Index const nβ = 2;
  Index const nT = std::floor((float)nS / nβ / nβ / nω);
  if (nT == 0) { Log::Fail("nT was zero"); }
  auto const R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const β1s = Eigen::ArrayXf::LinSpaced(nβ, β1lo, β1hi);
  auto const β2s = Eigen::ArrayXf::LinSpaced(nβ, β2lo, β2hi);
  auto const ωs = Eigen::ArrayXf::LinSpaced(nω, ωlo, ωhi);
  Log::Print("nT {} R1 {}:{} β1 {}:{} β2 {}:{} ω {}:{}", nT, R1lo, R1hi, β1lo, β1hi, β2lo, β2hi, ωlo, ωhi);

  Eigen::ArrayXXf p(4, nT * nβ * nβ * nω);

  Index ind = 0;
  for (Index iω = 0; iω < nω; iω++) {
    for (Index ib2 = 0; ib2 < nβ; ib2++) {
      for (Index ib1 = 0; ib1 < nβ; ib1++) {
        for (Index i1 = 0; i1 < nT; i1++) {
          p(0, ind) = 1.f / R1s(i1);
          p(1, ind) = β1s(ib1);
          p(2, ind) = β2s(ib2);
          p(3, ind) = ωs(iω);
          ind++;
        }
      }
    }
  }
  return p;
}

auto Simulate(Settings const settings, Eigen::ArrayXf const &p) -> Cx2
{
  float const R1 = 1.f / p(0);
  float const β1 = p(1);
  float const β2 = p(2);
  float const ω = p(3);

  Eigen::Matrix2f prep1, prep2;
  prep1 << β1, 0.f, 0.f, 1.f;
  prep2 << β2, 0.f, 0.f, 1.f;

  Eigen::Matrix2f E1, Eramp, Essi, Er, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state before first read-out
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp);
  Eigen::Matrix2f const SS =
    Essi * prep1 * grp.pow(settings.segsPerPrep - settings.segsPrep2) * Essi * prep2 * grp.pow(settings.segsPrep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(settings.spokesPerSeg * settings.segsKeep);
  for (Index ig = 0; ig < settings.segsPrep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = Essi * Erec * prep2 * Mz;
  for (Index ig = 0; ig < (settings.segsKeep - settings.segsPrep2); ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spokesPerSeg * settings.segsKeep) { Log::Fail("Programmer error"); }

  // Now do offresonance
  Cx1                       phase(settings.samplesPerSpoke);
  Eigen::VectorXcf::MapType pm(phase.data(), settings.samplesPerSpoke);
  auto const                sampPhase = settings.tSamp * ω;
  auto const                startPhase = settings.samplesInGap * sampPhase;
  auto const                endPhase = (settings.samplesInGap + settings.samplesPerSpoke - 1) * sampPhase;

  pm = Eigen::VectorXcf::LinSpaced(settings.samplesPerSpoke, startPhase * 1if, endPhase * 1if).array().exp();

  return phase.contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

auto Run(Settings const &s, Index const nsamp, std::vector<std::vector<float>> los, std::vector<std::vector<float>> his)
{
  if (los.size() != his.size()) { Log::Fail("Different number of parameter low bounds and high bounds"); }
  if (los.size() == 0) { Log::Fail("Must specify at least one set of tissue parameters"); }

  Index const     nP = 4;
  Eigen::ArrayXXf parameters(nP, nsamp * los.size());
  Log::Print("nsamp {} los {} parameters {} {}", nsamp, los.size(), parameters.rows(), parameters.cols());
  parameters.setZero();
  Index totalP = 0;
  for (size_t ii = 0; ii < los.size(); ii++) {
    Log::Print("Parameter set {}/{}. Low {} High {}", ii + 1, los.size(), fmt::join(los[ii], "/"), fmt::join(his[ii], "/"));
    auto p = T1β1β2ω(nsamp, los[ii], his[ii]);
    parameters.middleCols(totalP, p.cols()) = p;
    totalP += p.cols();
  }
  parameters.conservativeResize(nP, totalP);
  Log::Print("Parameters {} {}", parameters.rows(), parameters.cols());
  Cx3        dynamics(s.samplesPerSpoke, s.spokesPerSeg * s.segsKeep, totalP);
  auto const start = Log::Now();
  auto       task = [&](Index const ii) { dynamics.chip<2>(ii) = Simulate(s, parameters.col(ii)); };
  Threads::For(task, totalP, "Simulation");
  Log::Print("Simulation took {}", Log::ToNow(start));
  return dynamics;
}

void main_basis_sim2(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> sampPerSpoke(parser, "SAMP", "Samples per spoke", {"samp"}, 64);
  args::ValueFlag<Index> sampGap(parser, "GAP", "Samples in gap", {"gap"}, 2);
  args::ValueFlag<float> Tsamp(parser, "TS", "Sample time", {"tsamp"}, 10.e-6f);
  args::ValueFlag<Index> sps(parser, "SPS", "Spokes per segment", {'s', "sps"}, 128);
  args::ValueFlag<Index> spp(parser, "SPP", "Segments per prep", {'g', "spp"}, 1);
  args::ValueFlag<Index> sp2(parser, "G", "Segments before prep 2", {"sp2"}, 0);
  args::ValueFlag<Index> sk(parser, "K", "Segments to keep", {"sk"}, 1);
  args::ValueFlag<float> alpha(parser, "FLIP ANGLE", "Read-out fli,p-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float> ascale(parser, "A", "Flip-angle scaling", {"ascale"}, 1.);
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index> spoil(parser, "N", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<Index> k0(parser, "k0", "k0 navs", {"k0"}, 0);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float> Tprep(parser, "TPREP", "Time from prep to segment start", {"tprep"}, 0.f);
  args::ValueFlag<float> Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);

  args::ValueFlag<Index>     samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index>     gap(parser, "G", "Gap before samples begin", {"gap", 'g'}, 0);
  args::ValueFlag<Index>     tSamp(parser, "T", "Sample time (10μs)", {"tsamp", 't'}, 10);
  args::ValueFlagList<float> freqs(parser, "F", "Fat frequencies (-450 Hz)", {"freq", 'f'}, {440.f});

  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pLo(parser, "LO", "Low values for parameters",
                                                                                {"lo"});
  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pHi(parser, "HI", "High values for parameters",
                                                                                {"hi"});

  args::ValueFlag<Index> nsamp(parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 128);
  args::ValueFlag<Index> nBasis(parser, "N", "Number of basis vectors to retain (overrides threshold)", {"nbasis"}, 4);
  args::Flag             demean(parser, "C", "Mean-center dynamics", {"demean"});
  args::Flag             rotate(parser, "V", "Rotate basis", {"rotate"});
  args::Flag             normalize(parser, "N", "Normalize dynamics before SVD", {"normalize"});

  ParseCommand(parser);
  if (!oname) { throw args::Error("No output filename specified"); }

  Settings settings{
    .samplesPerSpoke = sampPerSpoke.Get(),
    .samplesInGap = sampGap.Get(),
    .spokesPerSeg = sps.Get(),
    .spokesSpoil = spoil.Get(),
    .segsPerPrep = spp.Get(),
    .segsPrep2 = sp2.Get(),
    .segsKeep = sk.Get(),
    .alpha = alpha.Get(),
    .ascale = ascale.Get(),
    .tSamp = Tsamp.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .Tprep = Tprep.Get(),
    .Trec = Trec.Get(),
  };

  Log::Print("nsamp {}", nsamp.Get());
  auto const                     dall = Run(settings, nsamp.Get(), pLo.Get(), pHi.Get());
  Eigen::ArrayXXcf::ConstMapType dmap(dall.data(), dall.dimension(0) * dall.dimension(1), dall.dimension(2));
  Log::Print("dmap {} {}", dmap.rows(), dmap.cols());
  SVDBasis<Cx> const             b(dmap, nBasis.Get(), demean, rotate, normalize);
  HD5::Writer                    writer(oname.Get());
  Log::Print("dall {} basis {} {}", dall.dimensions(), b.basis.rows(), b.basis.cols());
  writer.writeTensor(HD5::Keys::Basis, Sz3{b.basis.rows(), dall.dimension(0), dall.dimension(1)}, b.basis.data(),
                     HD5::Dims::Basis);
  writer.writeTensor(HD5::Keys::Dynamics, dall.dimensions(), dall.data(), {"sample", "trace", "p"});
  Log::Print("Finished {}", parser.GetCommand().Name());
}
