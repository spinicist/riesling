#pragma once

#include "log.h"
#include "trajectory.h"

#include <map>
#include <string>

namespace HD5 {

struct Writer
{
  Writer(std::string const &fname, Log &log_);
  ~Writer();
  void writeInfo(Info const &info);
  void writeMeta(std::map<std::string, float> const &meta);
  void writeTrajectory(Trajectory const &traj);
  void writeNoncartesian(Cx4 const &allVolumes);
  void writeCartesian(Cx4 const &volume);
  void writeImage(R4 const &volumes);
  void writeImage(Cx4 const &volumes);
  void writeSDC(R2 const &sdc);
  void writeSENSE(Cx4 const &sense);
  void writeBasis(R2 const &basis);
  void writeDynamics(R2 const &dynamics);
  void writeBasisImages(Cx5 const &basis);

  template <typename Scalar, int ND>
  void writeTensor(Eigen::Tensor<Scalar, ND> const &t, std::string const &label);
  void writeMatrix(Eigen::Ref<Eigen::MatrixXf const> const &m, std::string const &label);

private:
  Log &log_;
  int64_t handle_;
};

} // namespace HD5
