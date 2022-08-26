#include "rovir.hpp"

#include "algo/decomp.h"
#include "cropper.h"
#include "mapping.h"
#include "op/gridBase.hpp"
#include "op/nufft.hpp"
#include "sdc.h"
#include "tensorOps.h"

namespace rl {

ROVIROpts::ROVIROpts(args::Subparser &parser)
  : res(parser, "R", "ROVIR recon resolution", {"rovir-res"}, -1.f)
  , fov(parser, "F", "ROVIR Signal FoV", {"rovir-fov"}, -1.f)
  , loThresh(parser, "L", "ROVIR low threshold (percentile)", {"rovir-lo"}, 0.1f)
  , hiThresh(parser, "H", "ROVIR high threshold (percentile)", {"rovir-hi"}, 0.9f)
  , gap(parser, "G", "ROVIR FOV gap", {"rovir-gap"}, 0.f)
{
}

auto ROVIR(
  ROVIROpts &opts,
  Trajectory const &inTraj,
  float const energy,
  Index const channels,
  Index const lorestraces,
  Cx3 const &data) -> Eigen::MatrixXcf
{
  Index minRead = 0;
  Trajectory traj = inTraj;
  if (opts.res) {
    std::tie(traj, minRead) = inTraj.downsample(opts.res.Get(), lorestraces, true);
  }
  auto const &info = traj.info();
  Index const nC = info.channels;
  float const osamp = 3.f;
  auto gridder = make_grid<Cx>(traj, "FI3", osamp, info.channels);
  SDCOp sdc(SDC::Pipe(traj, true, osamp), nC);
  auto const sz = LastN<3>(gridder->inputDimensions());
  NUFFTOp nufft(sz, gridder.get(), &sdc);
  Cx4 const channelImages =
    nufft.adjoint(data.slice(Sz3{0, minRead, 0}, Sz3{nC, info.samples, info.traces})).chip<1>(0);

  // Get the signal distribution for thresholding
  Re3 const rss = ConjugateSum(channelImages, channelImages).real().sqrt(); // For ROI selection
  Log::Tensor(rss, "rovir-rss");
  std::vector<float> percentiles(rss.size());
  std::copy_n(rss.data(), rss.size(), percentiles.begin());
  std::sort(percentiles.begin(), percentiles.end());
  float const loVal = percentiles[(Index)std::floor(std::clamp(opts.loThresh.Get(), 0.f, 1.f) * (rss.size() - 1))];
  float const hiVal = percentiles[(Index)std::floor(std::clamp(opts.hiThresh.Get(), 0.f, 1.f) * (rss.size() - 1))];
  Log::Print(
    FMT_STRING("ROVIR signal thresholds {}-{}, full range {}-{}"),
    loVal,
    hiVal,
    percentiles.front(),
    percentiles.back());

  // Set up the masks

  Re3 signalMask(sz), interMask(sz);
  interMask = ((rss > loVal) && (rss < hiVal)).cast<float>();
  signalMask.setZero();
  Cropper sigCrop(info, sz, opts.fov.Get());
  sigCrop.crop3(signalMask) = sigCrop.crop3(interMask);
  Cropper intCrop(info, sz, opts.fov.Get() + opts.gap.Get());
  intCrop.crop3(interMask).setZero();

  Log::Tensor(signalMask, "rovir-signalmask");
  Log::Tensor(interMask, "rovir-interferencemask");

  // Copy to A & B matrices
  Index const nSig = Sum(signalMask);
  Index const nInt = Sum(interMask);
  Log::Print(FMT_STRING("{} voxels in signal mask, {} in interference mask"), nSig, nInt);

  Eigen::MatrixXcf Ω(nC, nSig);
  Eigen::MatrixXcf Γ(nC, nInt);
  Index isig = 0, iint = 0;
  for (Index iz = 0; iz < rss.dimension(2); iz++) {
    for (Index iy = 0; iy < rss.dimension(1); iy++) {
      for (Index ix = 0; ix < rss.dimension(0); ix++) {
        if (signalMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Ω(ic, isig) = channelImages(ic, ix, iy, iz);
          }
          isig++;
        }
        if (interMask(ix, iy, iz)) {
          for (Index ic = 0; ic < nC; ic++) {
            Γ(ic, iint) = channelImages(ic, ix, iy, iz);
          }
          iint++;
        }
      }
    }
  }

  Eigen::MatrixXcf const A = (Ω * Ω.adjoint()) / nSig;
  Eigen::MatrixXcf const B = (Γ * Γ.adjoint()) / nInt;
  Eigen::LLT<Eigen::MatrixXcf> const cholB(B);
  Eigen::MatrixXcf C = A.selfadjointView<Eigen::Lower>();
  cholB.matrixL().solveInPlace<Eigen::OnTheLeft>(C);
  cholB.matrixU().solveInPlace<Eigen::OnTheRight>(C);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(C);
  Log::Debug(FMT_STRING("A\n{}"), A);
  Log::Debug(FMT_STRING("B\n{}"), B);
  Log::Debug(FMT_STRING("C\n{}"), C);
  Log::Debug(FMT_STRING("eVals {}"), eig.eigenvalues().transpose());
  Log::Debug(FMT_STRING("eVecs\n{}"), eig.eigenvectors());
  Log::Debug(FMT_STRING("check\n{}"), eig.eigenvectors().colwise().norm());

  Eigen::ArrayXf vals = eig.eigenvalues().reverse().array().abs();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum /= *cumsum.end();
  Index nRetain = vals.rows();
  if ((energy > 0.f) && (energy <= 1.f)) {
    nRetain = (cumsum < energy).count();
  } else {
    nRetain = std::min(channels, vals.rows());
  }

  return eig.eigenvectors().rightCols(nRetain).rowwise().reverse();
}

} // namespace rl
