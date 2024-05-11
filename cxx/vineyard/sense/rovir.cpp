#include "sense/rovir.hpp"

#include "algo/decomp.hpp"
#include "algo/gs.hpp"
#include "algo/otsu.hpp"
#include "cropper.hpp"
#include "mapping.hpp"
#include "op/nufft.hpp"
#include "op/tensorscale.hpp"
#include "tensors.hpp"

namespace rl {

ROVIROpts::ROVIROpts(args::Subparser &parser)
  : res(parser, "R", "ROVIR calibration res (12 mm)", {"rovir-res"}, Eigen::Array3f::Constant(12.f))
  , fov(parser, "F", "ROVIR Signal FoV", {"rovir-fov"}, 1024.f)
  , gap(parser, "G", "ROVIR gap in voxels", {"rovir-gap"}, 0)
{
}

auto ROVIR(ROVIROpts        &opts,
           Trajectory const &inTraj,
           float const       energy,
           Index const       channels,
           Index const       lorestraces,
           Cx4 const        &inData) -> Eigen::MatrixXcf
{
  Trajectory traj = inTraj;
  Cx4        data;
  if (opts.res) {
    std::tie(traj, data) = inTraj.downsample(inData, opts.res.Get(), lorestraces, true, true);
  } else {
    data = inData;
  }
  Index const nC = data.dimension(0);
  float const osamp = 3.f;
  auto        nufft = make_nufft(traj, "ES3", osamp, nC, traj.matrixForFOV(opts.fov.Get()), IdBasis());
  auto const  sz = LastN<3>(nufft->ishape);
  Cx4 const   channelImages = nufft->adjoint(data).chip<1>(0);
  Re3 const   rss = ConjugateSum(channelImages, channelImages).real().sqrt().log1p(); // For ROI selection
  auto        thresh = Otsu(CollapseToArray(rss)).thresh;
  Log::Print("ROVIR signal threshold {}", thresh);
  Re3 signalMask(sz), interMask(sz);
  signalMask.setZero();
  interMask = (rss > thresh).cast<float>();
  Crop(signalMask, traj.matrix()) = Crop(interMask, traj.matrix());
  Sz3 szGap{traj.matrix()[0] + opts.gap.Get(), traj.matrix()[1] + opts.gap.Get(), traj.matrix()[2] + opts.gap.Get()};
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
  Index nRetain = vals.rows();
  if ((energy > 0.f) && (energy <= 1.f)) {
    nRetain = (cumsum < energy).count();
  } else {
    nRetain = std::min(channels, vals.rows());
  }
  Eigen::MatrixXcf vecs = eig.eigenvectors().array().rowwise().reverse().leftCols(nRetain);
  cholB.matrixU().solveInPlace(vecs);
  vecs = GramSchmidt(vecs);
  return vecs.leftCols(nRetain);
}

} // namespace rl
