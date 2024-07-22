#include "types.hpp"

#include "algo/decomp.hpp"
#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "sim/dir.hpp"
#include "sim/ir.hpp"
#include "sim/parameter.hpp"
#include "sim/prep.hpp"
#include "sim/t2flair.hpp"
#include "sim/t2prep.hpp"
#include "tensors.hpp"
#include "threads.hpp"

#include <Eigen/Householder>

using namespace rl;

template <typename T> auto Run(rl::Settings const &s, std::vector<Eigen::ArrayXf> plist)
{
  if (plist.size() == 0) { Log::Fail("Must specify at least one set of tissue parameters"); }

  T               seq{s};
  Index const     nP = T::nParameters;
  Eigen::ArrayXXf parameters(nP, plist.size());
  for (size_t ii = 0; ii < plist.size(); ii++) {
    Log::Print("Parameter set {}", fmt::streamed(plist[ii].transpose()));
    parameters.col(ii) = plist[ii];
  }
  Cx3        dynamics(parameters.cols(), s.samplesPerSpoke, seq.length());
  auto const start = Log::Now();
  auto       task = [&](Index const ii) { dynamics.chip<0>(ii) = seq.simulate(parameters.col(ii)); };
  Threads::For(task, parameters.cols(), "Simulation");
  Log::Print("Simulation took {}", Log::ToNow(start));
  return std::make_tuple(parameters, dynamics);
}

void main_basis_sim(args::Subparser &parser)
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

  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> plist(parser, "P", "Parameters", {"par"});

  args::Flag ortho(parser, "O", "Orthogonalize basis", {"ortho"});

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
  case Sequences::Prep: std::tie(pars, dall) = Run<rl::Prep>(settings, plist.Get()); break;
  case Sequences::Prep2: std::tie(pars, dall) = Run<rl::Prep2>(settings, plist.Get()); break;
  case Sequences::IR: std::tie(pars, dall) = Run<rl::IR>(settings, plist.Get()); break;
  case Sequences::IR2: std::tie(pars, dall) = Run<rl::IR2>(settings, plist.Get()); break;
  case Sequences::DIR: std::tie(pars, dall) = Run<rl::DIR>(settings, plist.Get()); break;
  case Sequences::T2Prep: std::tie(pars, dall) = Run<rl::T2Prep>(settings, plist.Get()); break;
  case Sequences::T2FLAIR: std::tie(pars, dall) = Run<rl::T2FLAIR>(settings, plist.Get()); break;
  }
  Sz3 const                 dshape = dall.dimensions();
  Index const               L = dshape[1] * dshape[2];
  Eigen::ArrayXXcf::MapType dmap(dall.data(), dshape[0], L);
  dmap.rowwise().normalize();

  if (ortho) {
    auto const             h = dmap.cast<Cxd>().matrix().transpose().householderQr();
    Eigen::MatrixXcd const I = Eigen::MatrixXcd::Identity(L, dshape[0]);
    Eigen::MatrixXcd const Q = h.householderQ() * I;
    Eigen::MatrixXcf const R = h.matrixQR().topRows(dshape[0]).cast<Cx>().triangularView<Eigen::Upper>();
    Eigen::MatrixXcf const Rinv = R.inverse();
    dmap = Q.transpose().cast<Cx>() * std::sqrt(L);
    Basis b(dall, Tensorfy(R, Sz2{R.rows(), R.cols()}));
    b.write(oname.Get());
  } else {
    dmap *= std::sqrt(L);
    Basis b(dall);
    b.write(oname.Get());
  }

  Log::Print("Finished {}", parser.GetCommand().Name());
}
