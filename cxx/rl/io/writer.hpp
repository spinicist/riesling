#pragma once

#include "../info.hpp"
#include "hd5-core.hpp"
#include <map>
#include <string>

namespace rl {
namespace HD5 {

struct Writer
{
  Writer(std::string const &fname, bool const append = false);
  ~Writer();
  void writeString(std::string const &label, std::string const &string);
  void writeStrings(std::string const &label, std::vector<std::string> const &string);
  void writeInfo(Info const &info);
  void writeTransform(Transform const &tfm, std::string const &lbl);
  void writeMeta(std::map<std::string, float> const &meta);

  template <typename Scalar, int N>
  void writeTensor(std::string const &label, Sz<N> const &shape, Scalar const *data, DimensionNames<N> const &dims);
  template <typename Derived> void writeMatrix(Eigen::DenseBase<Derived> const &m, std::string const &label);

  template <int N> void writeAttribute(std::string const &dataset, std::string const &attribute, Sz<N> const &val);

  bool exists(std::string const &name) const;

private:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
