#include "espirit.h"

#include "cropper.h"
#include "fft3n.h"
#include "gridder.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>

Cx4 ESPIRIT(
    Info const &info,
    R3 const &traj,
    float const os,
    Kernel *const kernel,
    long const calSz,
    long const kSz,
    float const retain,
    Cx3 const &data,
    Log &log)
{
  // Grid and heavily smooth each coil image, accumulate combined image
  float const sense_res = 6.f;
  log.info(FMT_STRING("ESPIRIT Calibration Size {} Kernel Size {}"), calSz, kSz);
  Gridder gridder(info, traj, os, SDC::Analytic, kernel, log, sense_res, false);
  Cx4 grid = gridder.newGrid();
  FFT3N fftGrid(grid, log);
  grid.setZero();
  gridder.toCartesian(data, grid);
  // Now reshape
  long const gridHalf = grid.dimension(1) / 2;
  long const calHalf = calSz / 2;
  long const kernelHalf = kSz / 2;
  long const nChan = grid.dimension(0);
  long const calTotal = calSz * calSz * calSz;
  Cx7 kernels(nChan, kSz, kSz, kSz, calSz, calSz, calSz);
  fmt::print(
      FMT_STRING("kernels {} grid {}\n"),
      fmt::join(kernels.dimensions(), ","),
      fmt::join(grid.dimensions(), ","));
  long const startPoint = gridHalf - calHalf - kernelHalf;
  for (long iz = 0; iz < calSz; iz++) {
    for (long iy = 0; iy < calSz; iy++) {
      for (long ix = 0; ix < calSz; ix++) {
        long const st_z = startPoint + iz;
        long const st_y = startPoint + iy;
        long const st_x = startPoint + ix;
        kernels.chip(iz, 6).chip(iy, 5).chip(ix, 4) =
            grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nChan, kSz, kSz, kSz});
      }
    }
  }
  Eigen::MatrixXcf kMat = CollapseToMatrix<Cx7, 4>(kernels);

  auto const bdcsvd = kMat.transpose().bdcSvd(Eigen::ComputeThinV);
  long const nRetain = std::lrintf(retain * calTotal);
  Cx5 kRetain(nChan, kSz, kSz, kSz, nRetain);
  auto retainMap = CollapseToMatrix<Cx5, 4>(kRetain);
  log.info(FMT_STRING("Retaining {} singular vectors"), nRetain);
  retainMap = bdcsvd.matrixV().leftCols(nRetain);

  // As the Matlab reference version says, "rotate kernel to order by maximum variance"
  auto retainReshaped = retainMap.reshaped(nChan, kSz * kSz * kSz * nRetain);
  Eigen::MatrixXcf rotation = retainReshaped.transpose().bdcSvd(Eigen::ComputeFullV).matrixV();

  fmt::print(
      FMT_STRING("rR {} {} rotation {} {}\n"),
      retainReshaped.rows(),
      retainReshaped.cols(),
      rotation.rows(),
      rotation.cols());
  retainMap = (retainReshaped.transpose() * rotation)
                  .transpose()
                  .reshaped(nChan * kSz * kSz * kSz, nRetain);

  Cx4 tempKernel(nChan, calSz, calSz, calSz);
  Log nullLog;
  FFT3N fftKernel(tempKernel, nullLog);
  Cx5 imgKernels(nChan, calSz, calSz, calSz, nRetain);
  for (long ii = 0; ii < nRetain; ii++) {
    ZeroPad(kRetain.chip(ii, 4), tempKernel);
    fftKernel.reverse();
    imgKernels.chip(ii, 4) = tempKernel;
  }

  Cx2 temp(nChan, nRetain);
  auto const tempMap = CollapseToMatrix(temp);
  Cx4 smallMaps(nChan, calSz, calSz, calSz);
  Cx1 oneVox(nChan);
  auto oneMap = CollapseToVector(oneVox);
  Cx4 eValues(nChan, calSz, calSz, calSz);
  eValues.setZero();
  Cx1 eVox(nChan);
  eVox.setZero();
  auto eMap = CollapseToVector(eVox);
  for (long iz = 0; iz < calSz; iz++) {
    for (long iy = 0; iy < calSz; iy++) {
      for (long ix = 0; ix < calSz; ix++) {
        temp = imgKernels.chip(iz, 3).chip(iy, 2).chip(ix, 1);
        auto const SVD = tempMap.bdcSvd(Eigen::ComputeFullU);
        eMap.real() = SVD.singularValues();
        eValues.chip(iz, 3).chip(iy, 2).chip(ix, 1) = eVox;

        Eigen::MatrixXcf U = SVD.matrixU();
        Eigen::ArrayXXcf const ph1 =
            (U.row(0).array().arg() * std::complex<float>(0.f, -1.f)).exp();
        Eigen::ArrayXXcf const ph = ph1.replicate(U.rows(), 1);
        Eigen::MatrixXcf const R = rotation * (U.array() * ph).matrix();
        oneMap = R.leftCols(1);
        smallMaps.chip(iz, 3).chip(iy, 2).chip(ix, 1) = oneVox;
      }
    }
  }

  WriteNifti(info, Cx4(eValues.shuffle(Sz4{1, 2, 3, 0})), "smallvalues.nii", log);
  WriteNifti(info, Cx4(smallMaps.shuffle(Sz4{1, 2, 3, 0})), "smallvectors.nii", log);
  // FFT, embed to full size, FFT again
  Cropper cropper(info, gridder.gridDims(), -1.f, log);
  FFT3N fftSmall(smallMaps, log);
  fftSmall.forward();
  ZeroPad(smallMaps, grid);
  fftGrid.reverse();

  log.info("Finished ESPIRIT");
  return grid;
}