#pragma once

#include "hd5-core.hpp"

#include "../info.hpp"

#include <map>
#include <string>

namespace rl {
namespace HD5 {
using Index = Eigen::Index;
struct IndexPair /* Simple class so we don't need std::pair*/
{
  Index dim, index;
};

/*
 * This class is for reading tensors out of generic HDF5 files. Used for SDC, SENSE maps, etc.
 */
struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname, bool const altComplex = false);
  Reader(Handle const fid, bool const altComplex = false);
  ~Reader();

  auto list(std::string const &id = "") const -> std::vector<std::string>;                 // List all datasets
  auto exists(std::string const &label = Keys::Data) const -> bool;                        // Does a data-set exist?
  auto exists(std::string const &dset, std::string const &attr) const -> bool;             // Check an attribute exists
  auto order(std::string const &label = Keys::Data) const -> Index;                        // Determine order of tensor dataset
  auto dimensions(std::string const &label = Keys::Data) const -> std::vector<Index>;      // Get Tensor dimensions
  auto listNames(std::string const &label = Keys::Data) const -> std::vector<std::string>; // Get dimension names

  auto readString(std::string const &label) const -> std::string; // Read a string dataset
  auto readStrings(std::string const &label) const -> std::vector<std::string>;
  auto readInfo() const -> Info; // Read the info struct from a file
  auto readTransform(std::string const &id) const -> Transform;
  auto readMeta() const -> std::map<std::string, float>; // Read meta-data group

  auto                  readAttributeFloat(std::string const &dataset, std::string const &attribute) const -> float;
  auto                  readAttributeInt(std::string const &dataset, std::string const &attribute) const -> long;
  template <int N> auto readAttributeSz(std::string const &dataset, std::string const &attribute) const -> Sz<N>;

  template <typename T> auto       readTensor(std::string const &label = Keys::Data) const -> T;
  template <typename T> void       readTo(T *data, std::string const &label = Keys::Data) const;
  template <int N> auto            dimensionNames(std::string const &label = Keys::Data) const -> DimensionNames<N>;
  template <typename T> auto       readSlab(std::string const &label, std::vector<IndexPair> const &chips) const -> T;
  template <typename Derived> auto readMatrix(std::string const &label) const -> Derived;

protected:
  Handle handle_;
  bool   owner_, altComplex_;
};

} // namespace HD5
} // namespace rl
