#include "log.h"
#include <algorithm>
#include <hdf5.h>

namespace HD5 {
using Handle = int64_t;

template <typename T>
struct type_tag
{
};

template <typename T>
hid_t type_impl(type_tag<T>);

template <typename T>
hid_t type()
{
  return type_impl(type_tag<T>{});
}

template <typename Scalar, int ND>
void store_tensor(
  Handle const &parent,
  std::string const &name,
  Eigen::Tensor<Scalar, ND> const &data,
  Log const &log)
{
  herr_t status;
  hsize_t ds_dims[ND], chunk_dims[ND];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(data.dimensions().rbegin(), ND, ds_dims);
  std::copy_n(ds_dims, ND, chunk_dims);
  if constexpr (ND > 3) {
    chunk_dims[0] = 1;
    // Try to stop chunk dimension going over 4 gig
    if (chunk_dims[1] > 1024) {
      chunk_dims[1] = 1024;
    }
  }

  auto const space = H5Screate_simple(ND, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, 2);
  status = H5Pset_chunk(plist, ND, chunk_dims);

  hid_t const tid = type<Scalar>();
  hid_t const dset = H5Dcreate(parent, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail("Could not write tensor {}, code: {}", name, status);
  } else {
    log.info("Wrote dataset: {}", name);
  }
}

template <int ND>
Eigen::DSizes<Index, ND> get_dims(Handle const &parent, std::string const &name, Log const &log)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail("Could not open tensor {}", name);
  }

  hid_t ds = H5Dget_space(dset);
  int const ndims = H5Sget_simple_extent_ndims(ds);
  if (ndims != ND) {
    Log::Fail("Number of dimensions {} does match on-disk number {}", ndims, ND);
  }
  std::array<hsize_t, ND> hdims;
  H5Sget_simple_extent_dims(ds, hdims.data(), NULL);
  Eigen::DSizes<Index, ND> dims;
  std::reverse_copy(hdims.begin(), hdims.end(), dims.begin());
  return dims;
}

template <typename Scalar, int ND>
void load_tensor(
  Handle const &parent, std::string const &name, Eigen::Tensor<Scalar, ND> &tensor, Log const &log)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor has '{}' has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND; ii++) {
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        FMT_STRING("Expected dimensions were {}, but were {} on disk"),
        fmt::join(tensor.dimensions(), ","),
        fmt::join(dims, ","));
    }
  }
  herr_t ret_value =
    H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail("Error reading tensor tensor {}, code: {}", name, ret_value);
  } else {
    log.info("Read dataset: {}", name);
  }
}

template <typename Scalar, int CD>
void load_tensor_slab(
  Handle const &parent,
  std::string const &name,
  Index const index,
  Eigen::Tensor<Scalar, CD> &tensor,
  Log const &log)
{
  int const ND = CD + 1;
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor has '{}' has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND - 1; ii++) {   // Last dimension is SUPPOSED to be different
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        FMT_STRING("Expected dimensions were {}, but were {} on disk"),
        fmt::join(tensor.dimensions(), ","),
        fmt::join(dims, ","));
    }
  }
  std::reverse(dims.begin(), dims.end()); // Reverse back

  std::array<hsize_t, ND> h5_start, h5_stride, h5_count, h5_block;
  h5_start[0] = index;
  std::fill_n(h5_start.begin() + 1, CD, 0);
  std::fill_n(h5_stride.begin(), ND, 1);
  std::fill_n(h5_count.begin(), ND, 1);
  h5_block[0] = 1;
  std::copy_n(dims.begin() + 1, CD, h5_block.begin() + 1);
  auto status = H5Sselect_hyperslab(
    ds, H5S_SELECT_SET, h5_start.data(), h5_stride.data(), h5_count.data(), h5_block.data());

  std::array<hsize_t, ND> mem_dims, mem_start, mem_stride, mem_count, mem_block;
  std::copy_n(dims.begin() + 1, CD, mem_dims.begin());
  std::fill_n(mem_start.begin(), CD, 0);
  std::fill_n(mem_stride.begin(), CD, 1);
  std::fill_n(mem_count.begin(), CD, 1);
  std::copy_n(dims.begin() + 1, CD, mem_block.begin());
  auto const mem_ds = H5Screate_simple(CD, mem_dims.data(), NULL);
  status = H5Sselect_hyperslab(
    mem_ds,
    H5S_SELECT_SET,
    mem_start.data(),
    mem_stride.data(),
    mem_count.data(),
    mem_block.data());

  status = H5Dread(dset, type<Scalar>(), mem_ds, ds, H5P_DEFAULT, tensor.data());
  if (status < 0) {
    Log::Fail("Error reading slab {} from tensor {}, code:", index, name, status);
  } else {
    log.info("Read dataset: {} chunk: {}", name, index);
  }
}

template <typename Scalar, int ND>
Eigen::Tensor<Scalar, ND> load_tensor(Handle const &parent, std::string const &name, Log const &log)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor has '{}' has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  typename Eigen::Tensor<Scalar, ND>::Dimensions tDims;
  std::copy_n(dims.begin(), ND, tDims.begin());
  std::reverse(tDims.begin(), tDims.end()); // HD5=row-major, Eigen=col-major
  Eigen::Tensor<Scalar, ND> tensor(tDims);
  herr_t ret_value =
    H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail("Error reading tensor tensor {}, code: {}", name, ret_value);
  } else {
    log.info("Read dataset: {}", name);
  }
  return tensor;
}

} // namespace HD5