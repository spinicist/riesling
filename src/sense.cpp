#include "sense.h"

#include "cropper.h"
#include "fft.h"
#include "gridder.h"
#include "io_nifti.h"
#include "log.h"
#include "threads.h"

Cx4 SENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    bool const stack,
    bool const kb,
    Cx3 const &data,
    Log &log)
{
  // Grid and heavily smooth each coil image, accumulate combined image
  float const sense_res = 6.f;
  log.info("Creating SENSE maps.");
  Gridder gridder(info, traj, os, true, kb, stack, log, sense_res, false);
  gridder.setDCExponent(0.5);
  Cx4 grid = gridder.newGrid();
  Cx3 rss = gridder.newGrid1();
  FFT3N fftN(grid, log);
  grid.setZero();
  rss.setZero();
  gridder.toCartesian(data, grid);
  fftN.reverse();
  rss.device(Threads::GlobalDevice()) = (grid * grid.conjugate()).sum(Sz1{0}).sqrt();
  WriteNifti(info, rss, "rss.nii", log);
  log.info("Normalizing channel images");
  grid = grid / tile(rss, info.channels);
  log.info("Finished SENSE maps");
  return grid;
}

Cx4 EigenSENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    bool const stack,
    bool const kb,
    long const nc,
    Cx3 const &data,
    Log &log)
{
  // Grid and heavily smooth each coil image, accumulate combined image
  float const sense_res = 6.f;
  log.info("Creating SENSE maps.");
  Gridder gridder(info, traj, os, true, kb, stack, log, sense_res, false);
  gridder.setDCExponent(0.5);
  Cx4 grid = gridder.newGrid();
  FFT3N fftN(grid, log);
  grid.setZero();
  gridder.toCartesian(data, grid);
  fftN.reverse();

  Cropper cropper(info, gridder.gridDims(), 50.f, stack, log);
  Cx4 cropped = cropper.crop4(grid);
  auto const cmap = CollapseToMatrix(cropped);
  auto const dm = cmap.colwise() - cmap.rowwise().mean();
  Eigen::MatrixXcf gramian = (dm.conjugate() * dm.transpose()) / pow(dm.norm(), 2);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(gramian);
  Eigen::ArrayXf const vals = eig.eigenvalues().reverse().array().abs();
  long const nkeep = std::clamp(nc, 0L, info.channels);
  log.info(
      FMT_STRING("Calculating reference from {} channels, total energy {}%"),
      nkeep,
      100.f * vals.head(nkeep).sum());
  Eigen::MatrixXcf evec = eig.eigenvectors().rightCols(nkeep).transpose();
  Cx3 emode = gridder.newGrid1();
  auto emap = CollapseToVector(emode);
  auto const gmap = CollapseToMatrix(grid);
  emap.noalias() = (evec * gmap).colwise().norm();

  WriteNifti(info, emode, "eigenmode.nii", log);

  log.info("Normalizing channel images");
  grid = grid / tile(emode, info.channels);
  log.info("Finished SENSE maps");
  return grid;
}
