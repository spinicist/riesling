#pragma once

#include "io/hd5-core.hpp"
#include "trajectory.hpp"
#include <map>
#include <string>

namespace rl {
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
  template <typename Derived>
  void writeMatrix(Eigen::DenseBase<Derived> const &m, std::string const &label);

  bool exists(std::string const &name) const;

private:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
