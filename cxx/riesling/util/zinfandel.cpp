#include "algo/admm.hpp"
#include "algo/lsqr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "prox/slr.hpp"
#include "threads.hpp"
#include "types.hpp"
#include "zin-grappa.hpp"
#include <filesystem>

using namespace rl;

void main_zinfandel(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  GridOpts gridOpts(parser);

  args::ValueFlag<Index> gap(parser, "G", "Set gap value (default 2)", {'g', "gap"}, 2);
  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 0)", {"lambda"}, 0.f);

  args::Flag             grappa(parser, "", "Use projection GRAPPA", {"grappa"});
  args::ValueFlag<Index> gSrc(parser, "S", "GRAPPA sources (default 4)", {"grappa-src"}, 4);
  args::ValueFlag<Index> gtraces(parser, "S", "GRAPPA calibration traces (default 4)", {"traces"}, 4);
  args::ValueFlag<Index> gRead(parser, "R", "GRAPPA calibration read samples (default 8)", {"read"}, 8);

  // SLR options
  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (2)", {"max-its"}, 2);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (30)", {"max-outer-its"}, 30);
  args::ValueFlag<float> ε(parser, "ABS", "ADMM tolerance (1e-4)", {"abs-tol"}, 1.e-4f);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM penalty parameter ρ (default 1)", {"rho"}, 1.f);

  args::ValueFlag<Index> kSz(parser, "SZ", "SLR Kernel Size (default 4)", {"kernel-size"}, 4);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);

  // Extend trajectory
  Re3 newPoints(traj.nDims(), gap.Get() + traj.nSamples(), traj.nTraces());
  newPoints.slice(Sz3{0, gap.Get(), 0}, Sz3{traj.nDims(), traj.nSamples(), traj.nTraces()}) = traj.points();
  Re3 dec = newPoints.slice(Sz3{0, gap.Get(), 0}, Sz3{traj.nDims(), 1, traj.nTraces()}) -
            newPoints.slice(Sz3{0, gap.Get() + 1, 0}, Sz3{traj.nDims(), 1, traj.nTraces()});
  for (Index ig = gap.Get(); ig > 0; ig--) {
    newPoints.slice(Sz3{0, ig - 1, 0}, Sz3{traj.nDims(), 1, traj.nTraces()}) =
      newPoints.slice(Sz3{0, ig, 0}, Sz3{traj.nDims(), 1, traj.nTraces()}) + dec;
  }
  Trajectory extended(newPoints, traj.matrix(), traj.voxelSize());
  Log::Print("Extended {}", extended.matrix());
  Cx5 const data = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(data.dimensions()));
  Index const nC = data.dimension(0);
  Index const nS = data.dimension(1);
  Index const nT = data.dimension(2);
  Index const nSlab = data.dimension(3);
  Index const nVol = data.dimension(4);
  Cx5         out{nC, nS + gap.Get(), nT, nSlab, nVol};

  if (grappa) {
    Cx3 vol{nC, nS + gap.Get(), nT};
    for (Index iv = 0; iv < nVol; iv++) {
      for (Index is = 0; is < nSlab; is++) {
        vol.slice(Sz3{0, gap.Get(), 0}, Sz3{nC, nS, nT}) = data.chip<4>(iv).chip<3>(is);
        zinGRAPPA(gap.Get(), gSrc.Get(), gtraces.Get(), gRead.Get(), λ.Get(), traj.points(), vol);
        out.chip<4>(iv).chip<3>(is) = vol;
      }
    }
  } else {
    // Use SLR
    auto const [lores, lo, sz] = traj.downsample(Eigen::Array3f{12.f, 12.f, 12.f}, 0, true, true);
    Log::Print("Extended {}", extended.matrix());
    auto      A = make_nufft(lores, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, lores.matrix());
    Sz5 const shape = A->ishape;
    Re2 const w = KSpaceSingle(lores);
    auto      M = std::make_shared<Ops::DiagRep<Cx>>(A->oshape[0], CollapseToArray(w).cast<Cx>());
    // auto M = std::make_shared<Ops::Identity<Cx>>(Product(shape));
    auto         id = std::make_shared<TensorIdentity<Cx, 5>>(shape);
    auto         slr = std::make_shared<Proxs::SLR>(λ.Get(), kSz.Get(), shape);
    ADMM::DebugX debug_x = [shape](Index const ii, ADMM::Vector const &x) {
      Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
    };
    ADMM::DebugZ debug_z = [shape](Index const ii, Index const ir, ADMM::Vector const &Fx, ADMM::Vector const &z,
                                   ADMM::Vector const &u) {
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), shape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), shape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), shape, u.data());
    };

    ADMM admm{A,       M,    {id}, {slr},   inner_its.Get(), 1, atol.Get(), btol.Get(), ctol.Get(), outer_its.Get(),
              ε.Get(), 1.2f, 10.f, debug_x, debug_z};

    Trajectory gapTraj(newPoints.slice(Sz3{0, 0, 0}, Sz3{traj.nDims(), gap.Get(), nT}), traj.matrix(), traj.voxelSize());
    auto       B = make_nufft(gapTraj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, lores.matrix());

    for (Index iv = 0; iv < nVol; iv++) {
      Cx4 d = data.chip<4>(iv).slice(Sz4{0, lo, 0, 0}, Sz4{nC, sz, nT, nSlab});
      Cx5 images = Tensorfy(admm.run(d.data(), ρ.Get()), A->ishape);
      Cx4 filled = B->forward(images);
      out.chip<4>(iv).slice(Sz3{0, 0, 0}, Sz3{nC, gap.Get(), nT}) = filled;
    }

    out.slice(Sz5{0, gap.Get(), 0, 0, 0}, Sz5{nC, nS, nT, nSlab, nVol}) = data;
  }

  HD5::Writer writer(coreOpts.oname.Get());
  Log::Print("Extended {}", extended.matrix());
  writer.writeInfo(reader.readInfo());
  extended.write(writer);
  writer.writeMeta(reader.readMeta());
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Noncartesian);
  Log::Print("Finished");
}
