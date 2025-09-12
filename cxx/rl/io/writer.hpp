#pragma once

#include "hd5-core.hpp"
#include <map>
#include <string>

namespace rl {
namespace HD5 {

void SetDeflate(Index const d); //! Set the global compression (deflate) level

struct Writer
{
  Writer(std::string const &fname, bool const append = false);
  ~Writer();
  void writeString(std::string const &label, std::string const &string);
  void writeStrings(std::string const &label, std::vector<std::string> const &string);
  void writeMeta(std::map<std::string, float> const &meta);

  template <typename T> void writeStruct(std::string const &lbl, T const &s) const;

  template <typename Scalar, size_t N> void
  writeTensor(std::string const &label, Shape<N> const &shape, Scalar const *data, DNames<N> const &dims);

  template <size_t N> void writeAttribute(std::string const &dataset, std::string const &attribute, Shape<N> const &val);

  bool exists(std::string const &name) const;

private:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
