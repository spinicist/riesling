#include "types.hpp"

#include "algo/decomp.hpp"
#include "basis/svd.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sim/dir.hpp"
#include "sim/ir.hpp"
#include "sim/parameter.hpp"
#include "sim/prep.hpp"
#include "sim/t2flair.hpp"
#include "sim/t2prep.hpp"
#include "threads.hpp"

using namespace rl;

template <typename T>
auto Run(rl::Settings const                &s,
         std::vector<Eigen::ArrayXf>        los,
         std::vector<Eigen::ArrayXf>        his,
         std::vector<Eigen::ArrayXf> const &Δs)
{
  if (los.size() != his.size()) { Log::Fail("Different number of parameter low bounds and high bounds"); }
  if (los.size() == 0) { Log::Fail("Must specify at least one set of tissue parameters"); }

  T               seq{s};
  Index const     nP = T::nParameters;
  Eigen::ArrayXXf parameters(nP, 0);
  for (size_t ii = 0; ii < los.size(); ii++) {
    Log::Print("Parameter set {}/{}. Low {} High {}", ii + 1, los.size(), fmt::join(los[ii], "/"), fmt::join(his[ii], "/"));
    auto p = ParameterGrid(nP, los[ii], his[ii], Δs[ii]);
    parameters.conservativeResize(nP, parameters.cols() + p.cols());
    parameters.rightCols(p.cols()) = p;
  }
  Cx3        dynamics(parameters.cols(), s.samplesPerSpoke, seq.length());
  auto const start = Log::Now();
  auto       task = [&](Index const ii) { dynamics.chip<0>(ii) = seq.simulate(parameters.col(ii)); };
  Threads::For(task, parameters.cols(), "Simulation");
  Log::Print("Simulation took {}", Log::ToNow(start));
  return std::make_tuple(parameters, dynamics);
}

void main_basis_svd(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::MapFlag<std::string, Sequences> seq(parser, "T", "Sequence type (default T1T2)", {"seq"}, SequenceMap);
  args::ValueFlag<Index>                samp(parser, "S", "Samples per spoke", {"samples"}, 64);
  args::ValueFlag<Index>                gap(parser, "G", "Samples in gap", {"gap"}, 3);
  args::ValueFlag<Index>                sps(parser, "SPS", "Spokes per segment", {'s', "sps"}, 128);
  args::ValueFlag<Index>                spp(parser, "SPP", "Segments per prep", {'g', "spp"}, 1);
  args::ValueFlag<Index>                sk(parser, "sk", "Segments per prep to keep", {"sk"}, 1);
  args::ValueFlag<Index>                sp2(parser, "G", "Segments before prep 2", {"sp2"}, 0);
  args::ValueFlag<float>                alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float>                ascale(parser, "A", "Flip-angle scaling", {"ascale"}, 1.);
  args::ValueFlag<float>                tsamp(parser, "T", "Sample time", {"tsamp"}, 10e-6);
  args::ValueFlag<float>                TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float>                Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index>                spoil(parser, "N", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<Index>                k0(parser, "k0", "k0 navs", {"k0"}, 0);
  args::ValueFlag<float>                Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float>                TI(parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.f);
  args::ValueFlag<float>                Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float>                te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);

  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> pLo(parser, "LO", "Low values for parameters", {"lo"});
  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> pHi(parser, "HI", "High values for parameters", {"hi"});
  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> pΔ(parser, "Δ", "Grid Δ for parameters", {"delta"});
  args::ValueFlag<Index> nRetain(parser, "N", "Number of basis vectors to retain (4)", {"nbasis"}, 4);

  args::Flag save(parser, "S", "Save dynamics and projections", {"save"});

  ParseCommand(parser);
  if (!oname) { throw args::Error("No output filename specified"); }

  rl::Settings settings{.samplesPerSpoke = samp.Get(),
                        .samplesGap = gap.Get(),
                        .spokesPerSeg = sps.Get(),
                        .spokesSpoil = spoil.Get(),
                        .k0 = k0.Get(),
                        .segsPerPrep = spp.Get(),
                        .segsKeep = sk ? sk.Get() : spp.Get(),
                        .segsPrep2 = sp2.Get(),
                        .alpha = alpha.Get(),
                        .ascale = ascale.Get(),
                        .Tsamp = tsamp.Get(),
                        .TR = TR.Get(),
                        .Tramp = Tramp.Get(),
                        .Tssi = Tssi.Get(),
                        .TI = TI.Get(),
                        .Trec = Trec.Get(),
                        .TE = te.Get()};
  Log::Print("{}", settings.format());

  Eigen::ArrayXXf pars;
  Cx3             dall;
  switch (seq.Get()) {
  case Sequences::Prep: std::tie(pars, dall) = Run<rl::Prep>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::Prep2: std::tie(pars, dall) = Run<rl::Prep2>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::IR: std::tie(pars, dall) = Run<rl::IR>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::IR2: std::tie(pars, dall) = Run<rl::IR2>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::DIR: std::tie(pars, dall) = Run<rl::DIR>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::T2Prep: std::tie(pars, dall) = Run<rl::T2Prep>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  case Sequences::T2FLAIR: std::tie(pars, dall) = Run<rl::T2FLAIR>(settings, pLo.Get(), pHi.Get(), pΔ.Get()); break;
  }
  Sz3 const                 dshape = dall.dimensions();
  Index const               L = dshape[1] * dshape[2];
  Eigen::MatrixXcf::MapType dmap(dall.data(), dshape[0], L);
  dmap.rowwise().normalize();
  Log::Print("Computing SVD {}x{}", dshape[0], L);
  SVD<Cx> svd(dmap);
  Log::Print("Variance explained: {}%", svd.variance(nRetain.Get()) * 100);

  Cx3                       basis(nRetain.Get(), dshape[1], dshape[2]);
  Eigen::MatrixXcf::MapType bmap(basis.data(), nRetain.Get(), L);
  bmap = svd.V.leftCols(nRetain.Get()).transpose();

  Log::Print("Computing projection");
  Cx3                       proj(dshape);
  Eigen::MatrixXcf::MapType pmap(proj.data(), dshape[0], L);
  Eigen::MatrixXcf const    temp = bmap.conjugate() * dmap.transpose();
  pmap = (bmap.transpose() * temp).transpose();

  bmap *= std::sqrt(L); // This is the correct scaling during the recon
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Basis, basis.dimensions(), basis.data(), HD5::Dims::Basis);
  if (save) {
    writer.writeTensor(HD5::Keys::Dynamics, dall.dimensions(), dall.data(), HD5::Dims::Basis);
    writer.writeTensor("projection", proj.dimensions(), proj.data(), HD5::Dims::Basis);
  }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
