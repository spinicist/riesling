#include "rl/basis/svd.hpp"
#include "rl/algo/decomp.hpp"
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

using namespace rl;

auto ParameterGrid(Index const nPar, Eigen::ArrayXf const &lo, Eigen::ArrayXf const &hi, Eigen::ArrayXi const &N)
  -> Eigen::ArrayXXf
{
  if (lo.size() != nPar) { throw Log::Failure("Pars", "Low values had {} elements, expected {}", lo.size(), nPar); }
  if (hi.size() != nPar) { throw Log::Failure("Pars", "High values had {} elements, expected {}", hi.size(), nPar); }
  if (N.size() != nPar) { throw Log::Failure("Pars", "N had {} elements, expected {}", N.size(), nPar); }

  Eigen::ArrayXf delta(nPar);
  Index          nTotal = 1;
  for (int ii = 0; ii < nPar; ii++) {
    if (N[ii] < 1) {
      throw Log::Failure("Pars", "{} N was less than 1", ii);
    } else if (N[ii] == 1) {
      delta[ii] = 0.f;
    } else {
      delta[ii] = (hi[ii] - lo[ii]) / (N[ii] - 1);
    }
    nTotal *= N[ii];
  }

  Eigen::ArrayXXf p(nPar, nTotal);
  Index           ind = 0;

  std::function<void(Index, Eigen::ArrayXf)> dimLoop = [&](Index dim, Eigen::ArrayXf pars) {
    for (Index id = 0; id < N[dim]; id++) {
      pars[dim] = lo[dim] + id * delta[dim];
      if (dim > 0) {
        dimLoop(dim - 1, pars);
      } else {
        p.col(ind++) = pars;
      }
    }
  };
  dimLoop(nPar - 1, Eigen::ArrayXf::Zero(nPar));
  return p;
}

auto Run(rl::SegmentedZTE const            &seq,
         std::vector<Eigen::ArrayXf> const &los,
         std::vector<Eigen::ArrayXf> const &his,
         std::vector<Eigen::ArrayXi> const &Ns)
{
  if (los.size() != his.size()) { throw Log::Failure("Sim", "Different number of parameter low bounds and high bounds"); }
  if (los.size() == 0) { throw Log::Failure("Sim", "Must specify at least one set of tissue parameters"); }

  Index const     nP = seq.nTissueParameters();
  Eigen::ArrayXXf parameters(nP, 0);
  for (size_t ii = 0; ii < los.size(); ii++) {
    auto const p = ParameterGrid(nP, los[ii], his[ii], Ns[ii]);
    Log::Print("Sim", "Parameter set {}/{}. Size {} Low {} High {}", ii + 1, los.size(), p.cols(), fmt::join(los[ii], "/"),
               fmt::join(his[ii], "/"));
    parameters.conservativeResize(nP, parameters.cols() + p.cols());
    parameters.rightCols(p.cols()) = p;
  }
  Log::Print("Sim", "Total parameter sets {}", parameters.cols());
  Cx3        dynamics(parameters.cols(), seq.samples(), seq.traces());
  auto const start = Log::Now();
  auto       task = [&](Index const ilo, Index const ihi) {
    for (Index ii = ilo; ii < ihi; ii++) {
      dynamics.chip<0>(ii) = seq.simulate(parameters.col(ii));
    }
  };
  Threads::ChunkFor(task, parameters.cols());
  Log::Print("Sim", "Simulation took {}. Final size {}", Log::ToNow(start), dynamics.dimensions());
  return std::make_tuple(seq.timepoints(), dynamics);
}

void main_basis_svd(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::MapFlag<std::string, Sequences> seq(parser, "T", "SegmentedZTE type (default T1T2)", {"seq"}, SequenceMap);
  args::ValueFlag<Index>                samp(parser, "S", "Samples per spoke", {"samples"}, 1);
  args::ValueFlag<Index>                gap(parser, "G", "Samples in gap", {"gap"}, 0);
  args::ValueFlag<Index>                sps(parser, "SPS", "Spokes per segment", {'s', "sps"}, 128);
  args::ValueFlag<Index>                spp(parser, "SPP", "Segments per prep", {'g', "spp"}, 1);
  args::ValueFlag<Index>                sp2(parser, "G", "Segments before prep 2", {"sp2"}, 0);
  args::ValueFlag<float>                alpha(parser, "FLIP ANGLE", "Read-out flip-angle", {'a', "alpha"}, 1.);
  args::ValueFlag<float>                ascale(parser, "A", "Flip-angle scaling", {"ascale"}, 1.);
  args::ValueFlag<float>                tsamp(parser, "T", "Sample time", {"tsamp"}, 10e-6);
  args::ValueFlag<float>                TR(parser, "TR", "Read-out repetition time", {"tr"}, 0.002f);
  args::ValueFlag<float>                Tramp(parser, "Tramp", "Ramp up/down times", {"tramp"}, 0.f);
  args::ValueFlag<Index>                spoil(parser, "SpoilTRs", "Spoil periods", {"spoil"}, 0);
  args::ValueFlag<Index>                k0(parser, "k0", "k0 navs", {"k0"}, 0);
  args::ValueFlag<float>                Tssi(parser, "Tssi", "Inter-segment time", {"tssi"}, 0.f);
  args::ValueFlag<float>                TI(parser, "TI", "Inversion time (from prep to segment start)", {"ti"}, 0.f);
  args::ValueFlag<float>                Trec(parser, "TREC", "Recover time (from segment end to prep)", {"trec"}, 0.f);
  args::ValueFlag<float>                te(parser, "TE", "Echo-time for MUPA/FLAIR", {"te"}, 0.f);
  args::Flag                            pt(parser, "P", "Pre-segment traces", {"pt"});
  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> pLo(parser, "LO", "Low values for parameters", {"lo"});
  args::ValueFlagList<Eigen::ArrayXf, std::vector, ArrayXfReader> pHi(parser, "HI", "High values for parameters", {"hi"});
  args::ValueFlagList<Eigen::ArrayXi, std::vector, ArrayXiReader> pN(parser, "N", "Grid N for parameters", {"N"});
  args::ValueFlag<Index> nRetain(parser, "N", "Number of basis vectors to retain (4)", {"nbasis"}, 4);
  args::Flag             norm(parser, "N", "Normalize dynamics", {"norm"});
  args::Flag             equalize(parser, "E", "Rotate basis to equalize variance", {"equalize"});
  args::Flag             scale(parser, "S", "Scale basis vectors by variance", {"scale"});
  args::Flag             save(parser, "S", "Save dynamics and projections", {"save"});

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
  case Sequences::ZTE: std::tie(time, dall) = Run(rl::SegmentedZTE(p, pt), pLo.Get(), pHi.Get(), pN.Get()); break;
  case Sequences::IR: std::tie(time, dall) = Run(rl::IR(p, pt), pLo.Get(), pHi.Get(), pN.Get()); break;
  case Sequences::DIR: std::tie(time, dall) = Run(rl::DIR(p, pt), pLo.Get(), pHi.Get(), pN.Get()); break;
  case Sequences::T2Prep: std::tie(time, dall) = Run(rl::T2Prep(p, pt), pLo.Get(), pHi.Get(), pN.Get()); break;
  case Sequences::T2FLAIR: std::tie(time, dall) = Run(rl::T2FLAIR(p, pt), pLo.Get(), pHi.Get(), pN.Get()); break;
  }
  Sz3 const   dshape = dall.dimensions();
  Index const L = dshape[1] * dshape[2];
  Index const N = nRetain.Get();

  Eigen::MatrixXcf::MapType dmap(dall.data(), dshape[0], L);

  if (norm) {
    Log::Print(cmd, "Normalizing entries");
    auto ntask = [&](Index const ilo, Index const ihi) {
      for (Index ii = ilo; ii < ihi; ii++) {
        dmap.row(ii) = dmap.row(ii).normalized();
      }
    };
    Threads::ChunkFor(ntask, dmap.rows());
  }

  Cx3                       basis(N, dshape[1], dshape[2]);
  Eigen::MatrixXcf::MapType bmap(basis.data(), N, L);

  Log::Print(cmd, "Computing SVD {}x{}", dmap.rows(), dmap.cols());
  SVD<Cxd> svd(dmap.cast<Cxd>());
  bmap = equalize ? svd.equalized(nRetain.Get()).cast<Cx>() : svd.basis(nRetain.Get(), scale.Get()).cast<Cx>();
  Log::Print(cmd, "Computing projection");
  Cx3                       proj(dshape);
  Eigen::MatrixXcf::MapType pmap(proj.data(), dshape[0], L);
  Eigen::MatrixXcf          temp = bmap.conjugate() * dmap.transpose();
  pmap = (bmap.transpose() * temp).transpose();
  auto resid = Norm<true>(dall - proj) / Norm<true>(dall);
  Log::Print(cmd, "Residual {}%", 100 * resid);
  HD5::Writer writer(oname.Get());
  writer.writeTensor(HD5::Keys::Basis, basis.dimensions(), basis.data(), HD5::Dims::Basis);
  writer.writeTensor("time", time.dimensions(), time.data(), {"t"});
  if (save) {
    writer.writeTensor(HD5::Keys::Dynamics, dall.dimensions(), dall.data(), HD5::Dims::Basis);
    writer.writeTensor("projection", proj.dimensions(), proj.data(), HD5::Dims::Basis);
  }
  Log::Print(cmd, "Finished");
}
