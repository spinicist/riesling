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

  template <typename Scalar, int N>
  void writeTensor(std::string const &label, Sz<N> const &shape, Scalar const *data, DimensionNames<N> const &dims = DimensionNames<N>());
  template <typename Derived>
  void writeMatrix(Eigen::DenseBase<Derived> const &m, std::string const &label);

  template <int N>
  void writeAttribute(std::string const &dataset, std::string const &attribute, Sz<N> const &val);

  bool exists(std::string const &name) const;

private:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
