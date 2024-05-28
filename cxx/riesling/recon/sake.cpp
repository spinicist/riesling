#include "algo/admm.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/compose.hpp"
#include "op/fft.hpp"
#include "op/hankel.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "prox/hermitian.hpp"
#include "prox/slr.hpp"
#include "scaling.hpp"

using namespace rl;

void main_sake(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  RlsqOpts   rlsqOpts(parser);

  args::ValueFlag<float> λ(parser, "L", "Regularization parameter (default 1e-1)", {"lambda"}, 1.e-1f);
  args::ValueFlag<Index> kSz(parser, "SZ", "SLR Kernel Size (default 5)", {"kernel-size"}, 5);

  args::ValueFlag<Eigen::Array3f, Array3fReader> ifov(parser, "FOV", "Iteration FOV (default 256,256,256)", {"ifov"},
                                                      Eigen::Array3f::Constant(256.f));

  args::Flag sep(parser, "S", "Separable kernels", {'s', "seperable"});
  args::Flag virtChan(parser, "V", "Use virtual conjugate channels", {"virtual"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nV = noncart.dimension(4);

  auto const  A = Recon::Channels(coreOpts.ndft, gridOpts, traj, ifov.Get(), nC, nS, basis);
  auto const  M = make_kspace_pre(traj, nC, basis, gridOpts.vcc, preOpts.type.Get(), preOpts.bias.Get());
  float const scale = Scaling(rlsqOpts.scaling, A, M, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);

  Sz5 const                shape = A->ishape;
  std::vector<Regularizer> regs;
  auto                     T = std::make_shared<TOps::Identity<Cx, 5>>(shape);
  if (sep) {
    auto sx = std::make_shared<Proxs::SLR<1>>(λ.Get(), shape, Sz1{2}, Sz1{kSz.Get()}, virtChan);
    auto sy = std::make_shared<Proxs::SLR<1>>(λ.Get(), shape, Sz1{3}, Sz1{kSz.Get()}, virtChan);
    auto sz = std::make_shared<Proxs::SLR<1>>(λ.Get(), shape, Sz1{4}, Sz1{kSz.Get()}, virtChan);
    regs.push_back({T, sx});
    regs.push_back({T, sy});
    regs.push_back({T, sz});
  } else {
    auto slr = std::make_shared<Proxs::SLR<3>>(λ.Get(), shape, Sz3{2, 3, 4}, Sz3{kSz.Get(), kSz.Get(), kSz.Get()}, virtChan);
    regs.push_back({T, slr});
  }

  ADMM::DebugX debug_x = [shape](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
  };
  ADMM::DebugZ debug_z = [shape = T->oshape](Index const ii, Index const ir, ADMM::Vector const &Fx, ADMM::Vector const &z,
                                             ADMM::Vector const &u) {
    Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), shape, Fx.data());
    Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), shape, z.data());
    Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), shape, u.data());
  };

  ADMM admm{A,
            M,
            regs,
            rlsqOpts.inner_its0.Get(),
            rlsqOpts.inner_its1.Get(),
            rlsqOpts.atol.Get(),
            rlsqOpts.btol.Get(),
            rlsqOpts.ctol.Get(),
            rlsqOpts.outer_its.Get(),
            rlsqOpts.ε.Get(),
            rlsqOpts.μ.Get(),
            rlsqOpts.τ.Get(),
            debug_x,
            debug_z};

  TOps::Pad<Cx, 5, 3> outFOV(traj.matrixForFOV(coreOpts.fov.Get()), A->ishape);
  Cx5                 out(AddBack(LastN<4>(outFOV.ishape), nV));
  for (Index iv = 0; iv < nV; iv++) {
    auto const channels = admm.run(&noncart(0, 0, 0, 0, iv), rlsqOpts.ρ.Get());
    auto const cropped = outFOV.adjoint(Tensorfy(channels, A->ishape));
    out.chip<4>(iv) = (cropped * cropped.conjugate()).sum(Sz1{0}).sqrt() / cropped.constant(scale);
  }
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Image);
  writer.writeInfo(info);
  writer.writeString("log", Log::Saved());
  Log::Print("Finished {}", parser.GetCommand().Name());
}
