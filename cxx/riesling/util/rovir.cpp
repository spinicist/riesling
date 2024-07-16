#include "algo/decomp.hpp"
#include "algo/gs.hpp"
#include "algo/otsu.hpp"
#include "algo/stats.hpp"
#include "basis/basis.hpp"
#include "compressor.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "inputs.hpp"
#include "tensors.hpp"
#include "types.hpp"

#include <Eigen/Eigenvalues>

using namespace rl;

void main_rovir(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  GridOpts gridOpts(parser);

  args::ValueFlag<Index> refVol(parser, "V", "Use this volume (default first)", {"vol"}, 0);
  args::ValueFlag<Index> lores(parser, "L", "Number of lores traces", {"lores"}, 0);
  args::ValueFlag<Index> nRetain(parser, "C", "Retain N channels (8)", {"channels"}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);

  args::ValueFlag<Eigen::Array3f, Array3fReader> res(parser, "R", "ROVIR calibration res (12 mm)", {"rovir-res"},
                                                     Eigen::Array3f::Constant(12.f));
  args::ValueFlag<float>                         fov(parser, "F", "ROVIR Signal FoV", {"rovir-fov"}, 1024.f);
  args::ValueFlag<Index>                         gap(parser, "G", "ROVIR gap in voxels", {"rovir-gap"}, 0);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  Cx4         data = reader.readSlab<Cx4>(HD5::Keys::Data, {{4, refVol.Get()}});
  if (res) { std::tie(traj, data) = traj.downsample(data, res.Get(), 0, true, true); }
  Index const nC = data.dimension(0);
  Index const nS = data.dimension(3);
  auto        nufft = Recon::Channels(coreOpts.ndft, gridOpts, traj, nC, nS, IdBasis());

  auto const sz = LastN<3>(nufft->ishape);
  Cx4 const  channelImages = nufft->adjoint(data).chip<1>(0);
  Re3 const  rss = ConjugateSum(channelImages, channelImages).real().sqrt().log1p(); // For ROI selection
  auto       thresh = Otsu(CollapseToArray(rss)).thresh;
  Log::Print("ROVIR signal threshold {}", thresh);
  Re3 signalMask(sz), interMask(sz);
  signalMask.setZero();
  interMask = (rss > thresh).cast<float>();
  Crop(signalMask, traj.matrix()) = Crop(interMask, traj.matrix());
  Sz3 szGap{traj.matrix()[0] + gap.Get(), traj.matrix()[1] + gap.Get(), traj.matrix()[2] + gap.Get()};
  Crop(interMask, szGap).setZero();
  // Copy to A & B matrices
  Index const nSig = Sum(signalMask);
  Index const nInt = Sum(interMask);
  Log::Print("{} voxels in signal mask, {} in interference mask", nSig, nInt);

  Eigen::MatrixXcf Ω = Eigen::MatrixXcf::Zero(nC, nSig);
  Eigen::MatrixXcf Γ = Eigen::MatrixXcf::Zero(nC, nInt);
  Index            isig = 0, iint = 0;
  for (Index iz = 0; iz < rss.dimension(2); iz++) {
    for (Index iy = 0; iy < rss.dimension(1); iy++) {
      for (Index ix = 0; ix < rss.dimension(0); ix++) {
        if (signalMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Ω(ic, isig) = channelImages(ic, ix, iy, iz);
          }
          isig++;
        } else if (interMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Γ(ic, iint) = channelImages(ic, ix, iy, iz);
          }
          iint++;
        }
      }
    }
  }
  Ω = Ω.colwise() - Ω.rowwise().mean();
  Γ = Γ.colwise() - Γ.rowwise().mean();

  Eigen::MatrixXcf A = (Ω.conjugate() * Ω.transpose()) / (nSig - 1);
  Eigen::MatrixXcf B = (Γ.conjugate() * Γ.transpose()) / (nInt - 1);
  Eigen::VectorXcf D = (A.diagonal().array().sqrt().inverse());
  A = D.asDiagonal() * A * D.asDiagonal();
  D = (B.diagonal().array().sqrt().inverse());
  B = D.asDiagonal() * B * D.asDiagonal();
  Eigen::LLT<Eigen::MatrixXcf> const cholB(B);
  Eigen::MatrixXcf                   C = A.selfadjointView<Eigen::Lower>();
  cholB.matrixL().solveInPlace<Eigen::OnTheLeft>(C);
  cholB.matrixU().solveInPlace<Eigen::OnTheRight>(C);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(C);

  Eigen::ArrayXf vals = eig.eigenvalues().array().reverse().abs();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum /= cumsum.tail(1)(0);
  Index n = vals.rows();
  if ((energy.Get() > 0.f) && (energy.Get() <= 1.f)) {
    n = (cumsum < energy.Get()).count();
  } else {
    n = std::min(nRetain.Get(), vals.rows());
  }
  Eigen::MatrixXcf vecs = eig.eigenvectors().array().rowwise().reverse().leftCols(n);
  cholB.matrixU().solveInPlace(vecs);
  auto const psi = GramSchmidt(vecs);

  HD5::Writer matfile(coreOpts.oname.Get());
  matfile.writeMatrix(psi, HD5::Keys::CompressionMatrix);
}
