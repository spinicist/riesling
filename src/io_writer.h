#pragma once

#include "log.h"
#include "trajectory.h"

#include <map>
#include <string>

namespace HD5 {

struct Writer
{
  Writer(std::string const &fname_);
  ~Writer();
  void writeInfo(Info const &info);
  void writeMeta(std::map<std::string, float> const &meta);
  void writeTrajectory(Trajectory const &traj);

  template <typename Scalar, int ND>
  void writeTensor(Eigen::Tensor<Scalar, ND> const &t, std::string const &label);
  void writeMatrix(Eigen::Ref<Eigen::MatrixXf const> const &m, std::string const &label);

private:
  int64_t handle_;
};

} // namespace HD5
