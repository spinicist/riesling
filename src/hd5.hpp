#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <hdf5.h>

namespace HD5 {

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
    Handle const &parent, std::string const &name, Eigen::Tensor<Scalar, ND> const &data)
{
  herr_t status;
  hsize_t ds_dims[ND], chunk_dims[ND];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(data.dimensions().rbegin(), ND, ds_dims);
  std::copy_n(ds_dims, ND, chunk_dims);

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
    throw std::runtime_error("Exception occurred while writing dataset " + std::to_string(status));
  }
}

template <typename T, int R, int C>
void store_array(Handle const &parent, std::string const &name, Eigen::Array<T, R, C> const &data)
{
  herr_t status;
  hsize_t ds_dims[2], chunk_dims[2];
  ds_dims[0] = data.cols();
  ds_dims[1] = data.rows();
  chunk_dims[0] = data.cols();
  chunk_dims[1] = data.rows();

  auto const space = H5Screate_simple(2, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, 2);
  status = H5Pset_chunk(plist, 2, chunk_dims);

  auto const type_id = type<T>();
  auto const dset =
      H5Dcreate(parent, name.c_str(), type_id, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(dset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    throw std::runtime_error("Exception occurred while writing dataset " + std::to_string(status));
  }
}

template <typename Scalar, int ND>
void load_tensor(Handle const &parent, std::string const &name, Eigen::Tensor<Scalar, ND> &tensor)
{
  using T = Eigen::Tensor<Scalar, ND>;
  hid_t dset = open_data(parent, name);
  hsize_t dims[ND];
  hid_t ds = H5Dget_space(dset);
  H5Sget_simple_extent_dims(ds, dims, NULL);
  typename T::Dimensions dimensions;
  for (int ii = 0; ii < ND; ii++) {
    // HD5=row-major, Eigen=col-major, so dimensions are reversed
    assert(dims[ii] == tensor.dimension(ND - ii));
  }
  herr_t ret_value =
      H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    throw std::runtime_error(fmt::format("Error reading tensor dataset: {}", name));
  }
}

template <typename T, int R, int C>
void load_array(Handle const &parent, std::string const &name, Eigen::Array<T, R, C> &array)
{
  hid_t dset = open_data(parent, name);
  hsize_t dims[2];
  hid_t ds = H5Dget_space(dset);
  H5Sget_simple_extent_dims(ds, dims, NULL);
  assert(array.cols() == dims[0]);
  assert(array.rows() == dims[1]);
  auto const type_id = type<T>();
  herr_t ret_value = H5Dread(dset, type_id, ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, array.data());
  if (ret_value < 0) {
    throw std::runtime_error(fmt::format("Error reading array dataset: {}", name));
  }
}

} // namespace HD5