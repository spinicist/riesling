#include "algo/decomp.h"
#include "compressor.h"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.h"
#include "op/grids.h"
#include "op/nufft.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "tensorOps.h"
#include "types.h"

#include <Eigen/Eigenvalues>

int main_compress(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);

  // One of the following must be set
  args::Flag pca(parser, "V", "Calculate PCA compression", {"pca"});
  args::Flag rovir(parser, "R", "Calculate ROVIR compression", {"rovir"});
  args::ValueFlag<std::string> ccFile(parser, "F", "Read compression matrix from file", {"cc-file"});

  // General options
  args::Flag save(parser, "S", "Save compression matrix to .h5 file", {"save"});
  args::ValueFlag<Index> channels(parser, "C", "Retain N channels (8)", {"channels"}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);
  args::ValueFlag<Index> refVol(parser, "V", "Use this volume (default last)", {"vol"});
  args::ValueFlag<Index> lores(parser, "L", "Number of lores spokes", {"lores"}, 0);

  // PCA Options
  args::ValueFlag<Sz2, Sz2Reader> pcaRead(parser, "R", "PCA Read Points (start, size)", {"pca-read"}, Sz2{0, 16});
  args::ValueFlag<Sz3, Sz3Reader> pcaSpokes(
    parser, "R", "PCA Spokes (start, size, stride)", {"pca-spokes"}, Sz3{0, 1024, 4});

  // ROVIR Options
  args::ValueFlag<float> res(parser, "R", "ROVIR recon resolution", {"rovir-res"}, -1.f);
  args::ValueFlag<float> fov(parser, "F", "ROVIR Signal FoV", {"rovir-fov"}, -1.f);
  args::ValueFlag<float> loThresh(parser, "L", "ROVIR low threshold (percentile)", {"rovir-lo"}, 0.1f);
  args::ValueFlag<float> hiThresh(parser, "H", "ROVIR high threshold (percentile)", {"rovir-hi"}, 0.9f);
  args::ValueFlag<float> gap(parser, "G", "ROVIR FOV gap", {"rovir-gap"}, 0.f);
  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto traj = reader.trajectory();
  auto const info = traj.info();
  Cx3 const ks = reader.noncartesian(ValOrLast(refVol, info.volumes));

  Compressor compressor;
  if (pca) {
    Sz2 const read = pcaRead.Get();
    Sz3 const spokes = pcaSpokes.Get();
    Index const maxRead = info.read_points - read[0];
    Index const nread = (read[1] > maxRead) ? maxRead : read[1];
    if (spokes[0] + spokes[1] > info.spokes) {
      Log::Fail(FMT_STRING("Requested end spoke {} is past end of file {}"), spokes[0] + spokes[1], info.spokes);
    }
    Log::Print(FMT_STRING("Using {} read points, {} spokes, {} stride"), nread, spokes[1], spokes[2]);
    Cx3 const ref =
      ks.slice(Sz3{0, read[0], spokes[0]}, Sz3{info.channels, read[1], spokes[1]}).stride(Sz3{1, 1, spokes[2]});

    auto const pc = PCA(CollapseToConstMatrix(ref), channels.Get(), energy.Get());
    compressor.psi = pc.vecs;
  } else if (rovir) {

    Index const nC = info.channels;
    auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
    Index minRead = 0;
    if (res) {
      std::tie(traj, minRead) = traj.downsample(res.Get(), lores.Get(), true);
    }
    Mapping const mapping(traj, kernel.get(), core.osamp.Get(), core.bucketSize.Get());
    auto gridder = make_grid(kernel.get(), mapping, info.channels, core.basisFile.Get());
    auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
    auto const sz = LastN<3>(gridder->inputDimensions());
    NUFFTOp nufft(sz, gridder.get(), sdc.get());
    Cx4 const channelImages =
      nufft.Adj(ks.slice(Sz3{0, minRead, 0}, Sz3{nC, traj.info().read_points, traj.info().spokes})).chip<1>(0);

    // Get the signal distribution for thresholding
    R3 const rss = ConjugateSum(channelImages, channelImages).real().sqrt(); // For ROI selection
    Log::Tensor(rss, "rovir-rss");
    std::vector<float> percentiles(rss.size());
    std::copy_n(rss.data(), rss.size(), percentiles.begin());
    std::sort(percentiles.begin(), percentiles.end());
    float const loVal = percentiles[(Index)std::floor(std::clamp(loThresh.Get(), 0.f, 1.f) * (rss.size() - 1))];
    float const hiVal = percentiles[(Index)std::floor(std::clamp(hiThresh.Get(), 0.f, 1.f) * (rss.size() - 1))];
    Log::Print(
      FMT_STRING("ROVIR signal thresholds {}-{}, full range {}-{}"),
      loVal,
      hiVal,
      percentiles.front(),
      percentiles.back());

    // Set up the masks

    R3 signalMask(sz), interMask(sz);
    interMask = ((rss > loVal) && (rss < hiVal)).cast<float>();
    signalMask.setZero();
    Cropper sigCrop(info, sz, fov.Get());
    sigCrop.crop3(signalMask) = sigCrop.crop3(interMask);
    Cropper intCrop(info, sz, fov.Get() + gap.Get());
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
    {
      compressor.psi = eig.eigenvectors().rowwise().reverse();
      Cx3 const rks = compressor.compress(ks);
      Cx4 const rImages = nufft.Adj(rks).chip<1>(0);
      Log::Tensor(channelImages, "rovir-channels");
      Log::Tensor(rImages, "rovir-rchannels");
    }

    Eigen::ArrayXf vals = eig.eigenvalues().reverse().array().abs();
    Eigen::ArrayXf cumsum(vals.rows());
    std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
    cumsum /= *cumsum.end();
    Index nRetain = vals.rows();
    if ((energy.Get() > 0.f) && (energy.Get() <= 1.f)) {
      nRetain = (cumsum < energy.Get()).count();
    } else {
      nRetain = std::min(channels.Get(), vals.rows());
    }
    compressor.psi = eig.eigenvectors().rightCols(nRetain).rowwise().reverse();
  } else if (ccFile) {
    HD5::Reader matFile(ccFile.Get());
    compressor.psi = matFile.readMatrix<Eigen::MatrixXcf>(HD5::Keys::CompressionMatrix);
  } else {
    Log::Fail("Must specify PCA/ROVIR/load from file");
  }
  Cx4 all_ks = info.noncartesianSeries();
  for (Index iv = 0; iv < info.volumes; iv++) {
    all_ks.chip<3>(iv) = reader.noncartesian(iv);
  }
  Info out_info = info;
  out_info.channels = compressor.out_channels();
  Cx4 out_ks = out_info.noncartesianSeries();
  compressor.compress(all_ks, out_ks);

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "compressed"));
  writer.writeTrajectory(Trajectory(out_info, traj.points(), traj.frames()));
  writer.writeTensor(out_ks, HD5::Keys::Noncartesian);

  if (save) {
    HD5::Writer matfile(OutName(core.iname.Get(), core.oname.Get(), "ccmat"));
    matfile.writeMatrix(compressor.psi, HD5::Keys::CompressionMatrix);
  }
  return EXIT_SUCCESS;
}
