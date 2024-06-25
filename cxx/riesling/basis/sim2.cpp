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
  float tSamp = 10e-6, alpha = 1.f, TR = 2.e-3f, Tramp = 10.e-3f, Tssi = 10.e-3f, Tprep = 0, Trec = 0;
};

auto T1β1β2ω(std::vector<float> lo, std::vector<float> hi, std::vector<float> const spacing) -> Eigen::ArrayXXf
{
  if (spacing.size() != 4) { Log::Fail("Spacing had wrong number of elements"); }
  float const R1lo = 1.f / lo[0];
  float const R1hi = 1.f / hi[0];
  float const R1Δ = spacing[0];
  float const β1lo = lo[1];
  float const β1hi = hi[1];
  float const β1Δ = spacing[1];
  float const β2lo = lo[2];
  float const β2hi = hi[2];
  float const β2Δ = spacing[2];
  float const f0lo = lo[3];
  float const f0hi = hi[3];
  float const f0Δ = spacing[3];
  Index const nf0 = std::max<Index>(1, (f0hi - f0lo) / f0Δ);
  Index const nβ1 = std::max<Index>(1, (β1hi - β1lo) / β1Δ);
  Index const nβ2 = std::max<Index>(1, (β2hi - β2lo) / β2Δ);
  Index const nR1 = std::max<Index>(1, (R1hi - R1lo) / R1Δ);
  auto const  R1s = Eigen::ArrayXf::LinSpaced(nR1, R1lo, R1hi);
  auto const  β1s = Eigen::ArrayXf::LinSpaced(nβ1, β1lo, β1hi);
  auto const  β2s = Eigen::ArrayXf::LinSpaced(nβ2, β2lo, β2hi);
  auto const  f0s = Eigen::ArrayXf::LinSpaced(nf0, f0lo, f0hi);
  Log::Print("R1 {} {}:{} β1 {} {}:{} β2 {} {}:{} f0 {} {}:{}", nR1, R1lo, R1hi, nβ1, β1lo, β1hi, nβ2, β2lo, β2hi, nf0, f0lo, f0hi);

  Eigen::ArrayXXf p(4, nR1 * nβ1 * nβ2 * nf0);

  Index ind = 0;
  for (Index if0 = 0; if0 < nf0; if0++) {
    for (Index ib2 = 0; ib2 < nβ1; ib2++) {
      for (Index ib1 = 0; ib1 < nβ2; ib1++) {
        for (Index i1 = 0; i1 < nR1; i1++) {
          p(0, ind) = 1.f / R1s(i1);
          p(1, ind) = β1s(ib1);
          p(2, ind) = β2s(ib2);
          p(3, ind) = f0s(if0);
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
  float const f0 = p(3);

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
  float const               sampPhase = settings.tSamp * f0 * 2 * M_PI;
  float const               startPhase = settings.samplesInGap * sampPhase;
  float const               endPhase = (settings.samplesInGap + settings.samplesPerSpoke - 1) * sampPhase;
  pm = Eigen::VectorXcf::LinSpaced(settings.samplesPerSpoke, startPhase * 1if, endPhase * 1if).array().exp();
  Log::Print("Phase sample {} start {} {} end {} {}", sampPhase, startPhase, std::arg(pm[0]), endPhase, std::arg(pm[settings.samplesPerSpoke - 1]));
  return phase.contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

auto Run(Settings const                 &s,
         std::vector<std::vector<float>> los,
         std::vector<std::vector<float>> his,
         std::vector<float>              spacings)
{
  if (los.size() != his.size()) { Log::Fail("Different number of parameter low bounds and high bounds"); }
  if (los.size() == 0) { Log::Fail("Must specify at least one set of tissue parameters"); }

  Index const     nP = 4;
  Eigen::ArrayXXf parameters(nP, 0);
  Log::Print("los {} parameters {} {}", los.size(), parameters.rows(), parameters.cols());
  parameters.setZero();
  for (size_t ii = 0; ii < los.size(); ii++) {
    Log::Print("Parameter set {}/{}. Low {} High {}", ii + 1, los.size(), fmt::join(los[ii], "/"), fmt::join(his[ii], "/"));
    auto p = T1β1β2ω(los[ii], his[ii], spacings);
    parameters.conservativeResize(4, parameters.cols() + p.cols());
    parameters.rightCols(p.cols()) = p;
  }
  Log::Print("Parameters {} {}", parameters.rows(), parameters.cols());
  Cx3        dynamics(s.samplesPerSpoke, s.spokesPerSeg * s.segsKeep, parameters.cols());
  auto const start = Log::Now();
  auto       task = [&](Index const ii) { dynamics.chip<2>(ii) = Simulate(s, parameters.col(ii)); };
  Threads::For(task, parameters.cols(), "Simulation");
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
  args::ValueFlag<float> TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float> Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index> spoil(parser, "N", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<Index> k0(parser, "k0", "k0 navs", {"k0"}, 0);
  args::ValueFlag<float> Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float> Tprep(parser, "TPREP", "Time from prep to segment start", {"tprep"}, 0.f);
  args::ValueFlag<float> Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);

  args::ValueFlag<Index> gap(parser, "G", "Gap before samples begin", {"gap", 'g'}, 0);
  args::ValueFlag<Index> tSamp(parser, "T", "Sample time (10μs)", {"tsamp", 't'}, 10);

  args::ValueFlag<std::vector<float>, VectorReader<float>> spacings(parser, "S", "Parameter spacings", {"spacing"},
                                                                    {1.f, 1.f, 1.f, 1.f});
  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pLo(parser, "LO", "Low values for parameters",
                                                                                {"lo"});
  args::ValueFlagList<std::vector<float>, std::vector, VectorReader<float>> pHi(parser, "HI", "High values for parameters",
                                                                                {"hi"});

  args::ValueFlag<Index> nsamp(parser, "N", "Number of samples per tissue (default 2048)", {"nsamp"}, 128);
  args::Flag             svd(parser, "S", "Do SVD", {"svd"});
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
    .tSamp = Tsamp.Get(),
    .alpha = alpha.Get(),
    .TR = TR.Get(),
    .Tramp = Tramp.Get(),
    .Tssi = Tssi.Get(),
    .Tprep = Tprep.Get(),
    .Trec = Trec.Get(),
  };

  Log::Print("nsamp {}", nsamp.Get());
  auto                      dall = Run(settings, pLo.Get(), pHi.Get(), spacings.Get());
  Eigen::ArrayXXcf::MapType dmap(dall.data(), dall.dimension(0) * dall.dimension(1), dall.dimension(2));

  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Dynamics, dall.dimensions(), dall.data(), {"sample", "trace", "p"});
  if (svd) {
    SVDBasis<Cx> const b(dmap, nBasis.Get(), demean, rotate, normalize);
    writer.writeTensor(HD5::Keys::Basis, Sz3{b.basis.rows(), dall.dimension(0), dall.dimension(1)}, b.basis.data(),
                       HD5::Dims::Basis);
  } else {
    if (normalize) {dmap = dmap.colwise().normalized() * std::sqrt(dmap.rows()); }
    Cx3 const shuffled = dall.shuffle(Sz3{2, 0, 1});
    writer.writeTensor(HD5::Keys::Basis, shuffled.dimensions(), shuffled.data(), HD5::Dims::Basis);
  }

  Log::Print("Finished {}", parser.GetCommand().Name());
}
