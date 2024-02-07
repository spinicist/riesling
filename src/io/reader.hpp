#pragma once

#include "io/hd5-core.hpp"

#include "info.hpp"

#include <map>
#include <string>

namespace rl {

namespace HD5 {

/*
 * This class is for reading tensors out of generic HDF5 files. Used for SDC, SENSE maps, etc.
 */
struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname);
  ~Reader();

  auto list() const -> std::vector<std::string>;       // List all datasets
  auto exists(std::string const &label) const -> bool; // Does a data-set exist?
  auto order(std::string const &label) const -> Index;  // Determine order of tensor dataset
  auto dimensions(std::string const &label) const -> std::vector<Index>; // Get Tensor dimensions
  auto readInfo() const -> Info;                                                 // Read the info struct from a file
  auto readMeta() const -> std::map<std::string, float>;                         // Read meta-data group

  template <typename T>
  auto readAttribute(std::string const &dataset, std::string const &attribute) -> T;

  template <typename T>
  auto readTensor(std::string const &label) const -> T;
  template <int N>
  auto readDims(std::string const &label) const -> Names<N>;
  template <typename T>
  auto readSlab(std::string const &label, std::vector<Index> const &sliceDims, std::vector<Index> const &sliceInds) const -> T;
  template <typename Derived>
  auto readMatrix(std::string const &label) const -> Derived;

protected:
  Handle handle_;
};

} // namespace HD5
} // namespace rl
