#include "log.h"
#include <algorithm>
#include <hdf5.h>

namespace HD5 {
using Handle = uint64_t;

template <typename T>
struct type_tag
{
};

hid_t type_impl(type_tag<float>)
{
  return H5T_NATIVE_FLOAT;
}

hid_t type_impl(type_tag<double>)
{
  return H5T_NATIVE_DOUBLE;
}

template <typename T>
hid_t type_impl(type_tag<std::complex<T>>)
{
  struct complex_t
  {
    T r; /*real part*/
    T i; /*imaginary part*/
  };

  hid_t scalar_id = type_impl(type_tag<T>{});
  hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
  herr_t status;
  status = H5Tinsert(complex_id, "r", HOFFSET(complex_t, r), scalar_id);
  status = H5Tinsert(complex_id, "i", HOFFSET(complex_t, i), scalar_id);
  if (status) {
    throw std::runtime_error(
        "Exception occurred creating complex datatype " + std::to_string(status));
  }
  return complex_id;
}

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
    log.fail("Could not write tensor {}, code: {}", name, status);
  } else {
    log.info("Wrote dataset: {}", name);
  }
}

template <typename Scalar, int ND>
void load_tensor(
    Handle const &parent,
    std::string const &name,
    Eigen::Tensor<Scalar, ND> &tensor,
    Log const &log)
{
  using T = Eigen::Tensor<Scalar, ND>;
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    log.fail("Could not open tensor {}", name);
  }
  hsize_t dims[ND];
  hid_t ds = H5Dget_space(dset);
  H5Sget_simple_extent_dims(ds, dims, NULL);
  typename T::Dimensions dimensions;
  for (int ii = 0; ii < ND; ii++) {
    // HD5=row-major, Eigen=col-major, so dimensions are reversed
    assert(dims[ii] == tensor.dimension(ND - 1 - ii));
  }
  herr_t ret_value =
      H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    log.fail("Error reading tensor tensor {}, code: {}", name, ret_value);
  } else {
    log.info("Read dataset: {}", name);
  }
}

template <typename Scalar, int CD>
void load_tensor_slab(
    Handle const &parent,
    std::string const &name,
    long const index,
    Eigen::Tensor<Scalar, CD> &tensor,
    Log const &log)
{
  int const ND = CD + 1;
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    log.fail("Could not open tensor {}", name);
  }
  hsize_t dims[ND];
  hid_t ds = H5Dget_space(dset);
  H5Sget_simple_extent_dims(ds, dims, NULL);
  fmt::print("dims {} tensor {}\n", fmt::join(dims, ","), fmt::join(tensor.dimensions(), ","));
  for (int ii = 1; ii < ND; ii++) {
    // HD5=row-major, Eigen=col-major, so dimensions are reversed
    assert(dims[ii] == tensor.dimension(CD - ii));
  }

  hsize_t start[ND], stride[ND], count[ND], block[ND];
  std::fill_n(&start[1], CD, 0);
  start[0] = index;
  std::fill_n(&stride[0], ND, 1);
  std::fill_n(&count[0], ND, 1);
  std::copy_n(&dims[1], CD, &block[1]);
  block[0] = 1;

  fmt::print(
      "start {} stride {} count {} block {}\n",
      fmt::join(start, ","),
      fmt::join(stride, ","),
      fmt::join(count, ","),
      fmt::join(block, ","));

  herr_t status = H5Sselect_hyperslab(ds, H5S_SELECT_SET, start, stride, count, block);
  status = H5Dread(dset, type<Scalar>(), H5S_ALL, ds, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (status < 0) {
    log.fail("Error reading slab {} from tensor {}, code:", index, name, status);
  } else {
    log.info("Read dataset: {} chunk: {}", name, index);
  }
}

} // namespace HD5