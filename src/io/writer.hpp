#pragma once

#include "info.hpp"
#include "io/hd5-core.hpp"
#include <map>
#include <string>

namespace rl {
namespace HD5 {

struct Writer
{
  Writer(std::string const &fname_);
  ~Writer();
  void writeString(std::string const &label, std::string const &string);
  void writeInfo(Info const &info);
  void writeMeta(std::map<std::string, float> const &meta);

  template <typename Scalar, int ND>
  void writeTensor(std::string const &label, Sz<ND> const &shape, Scalar const *data);
  template <typename Derived>
  void writeMatrix(Eigen::DenseBase<Derived> const &m, std::string const &label);

  bool exists(std::string const &name) const;

private:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
