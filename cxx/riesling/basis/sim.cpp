#include "rl/algo/decomp.hpp"
#include "rl/basis/basis.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/sim/dir.hpp"
#include "rl/sim/ir.hpp"
#include "rl/sim/t2flair.hpp"
#include "rl/sim/t2prep.hpp"
#include "rl/sim/zte.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

#include <Eigen/Householder>

using namespace rl;

auto Run(rl::SegmentedZTE const &seq, std::vector<Eigen::ArrayXf> plist)
{
  if (plist.size() == 0) { throw Log::Failure("Sim", "Must specify at least one set of tissue parameters"); }

  Index const     nP = seq.nTissueParameters();
  Eigen::ArrayXXf parameters(nP, plist.size());
  for (size_t ii = 0; ii < plist.size(); ii++) {
    Log::Print("Sim", "Parameter set {}", fmt::streamed(plist[ii].transpose()));
    parameters.col(ii) = plist[ii];
  }
  Cx3        dynamics(parameters.cols(), seq.samples(), seq.traces());
  auto const start = Log::Now();
  auto       task = [&](Index const ilo, Index const ihi) {
    for (Index ii = ilo; ii < ihi; ii++) {
      dynamics.chip<0>(ii) = seq.simulate(parameters.col(ii));
    }
  };
  Threads::ChunkFor(task, parameters.cols());
  Log::Print("Sim", "Simulation took {}", Log::ToNow(start));
  return std::make_tuple(seq.timepoints(), dynamics);
}

void main_basis_sim(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::MapFlag<std::string, Sequences> seq(parser, "T", "Sequence type (default ZTE)", {"seq"}, SequenceMap);
  args::ValueFlag<Index>                samp(parser, "S", "Samples per spoke", {"samples"}, 64);
  args::ValueFlag<Index>                gap(parser, "G", "Samples in gap", {"gap"}, 3);
  args::ValueFlag<Index>                sps(parser, "SPS", "Spokes per segment", {'s', "sps"}, 128);
  args::ValueFlag<Index>                spp(parser, "SPP", "Segments per prep", {'g', "spp"}, 1);
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
  args::Flag                            pt(parser, "P", "Pre-segment traces", {"pt"});
  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> plist(parser, "P", "Pars", {"par"});

  args::Flag norm(parser, "N", "Normalize basis", {"norm", 'n'});
  args::Flag ortho(parser, "O", "Orthogonalize basis", {"ortho"});

  ParseCommand(parser);
  auto const cmd = parser.GetCommand().Name();
  if (!oname) { throw args::Error("No output filename specified"); }

  rl::SegmentedZTE::Pars p{.samplesPerSpoke = samp.Get(),
                           .samplesGap = gap.Get(),
                           .spokesPerSeg = sps.Get(),
                           .spokesSpoil = spoil.Get(),
                           .k0 = k0.Get(),
                           .segsPerPrep = spp.Get(),
                           .segsPrep2 = sp2.Get(),
                           .alpha = alpha.Get() * (float)M_PI / 180.f,
                           .ascale = ascale.Get(),
                           .Tsamp = tsamp.Get(),
                           .TR = TR.Get(),
                           .Tramp = Tramp.Get(),
                           .Tssi = Tssi.Get(),
                           .TI = TI.Get(),
                           .Trec = Trec.Get(),
                           .TE = te.Get()};
  Log::Print(cmd, "{}", p.format());

  Re1 time;
  Cx3 dall;
  switch (seq.Get()) {
  case Sequences::ZTE: std::tie(time, dall) = Run(rl::SegmentedZTE(p, pt), plist.Get()); break;
  case Sequences::IR: std::tie(time, dall) = Run(rl::IR(p, pt), plist.Get()); break;
  case Sequences::DIR: std::tie(time, dall) = Run(rl::DIR(p, pt), plist.Get()); break;
  case Sequences::T2Prep: std::tie(time, dall) = Run(rl::T2Prep(p, pt), plist.Get()); break;
  case Sequences::T2FLAIR: std::tie(time, dall) = Run(rl::T2FLAIR(p, pt), plist.Get()); break;
  }
  Sz3 const                 dshape = dall.dimensions();
  Index const               M = dshape[0];
  Index const               N = dshape[1] * dshape[2];
  Eigen::ArrayXXcf::MapType dmap(dall.data(), M, N);

  if (norm) { dmap.rowwise().normalize(); }
  if (ortho) {
    auto const             h = dmap.cast<Cxd>().matrix().transpose().householderQr();
    Eigen::MatrixXcd const I = Eigen::MatrixXcd::Identity(N, M);
    Eigen::MatrixXcf Q = (h.householderQ() * I).cast<Cx>();
    Eigen::MatrixXcf R = h.matrixQR().topRows(M).cast<Cx>().triangularView<Eigen::Upper>();
    if ((R.diagonal().array().real() < 0.f).all()) { /* Flip so things are positive */
      Q = -Q;
      R = -R;
    }
    dmap = Q.transpose().cast<Cx>();
    Basis b(dall, AsTensorMap(R, Sz2{R.rows(), R.cols()}));
    b.write(oname.Get());
  } else {
    Basis b(dall);
    b.write(oname.Get());
  }

  Log::Print(cmd, "Finished");
}
