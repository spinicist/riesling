#include "io/reader.hpp"

#include "log.hpp"
#include <filesystem>
#include <hdf5.h>

namespace rl {
namespace HD5 {

Index getRank(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor {}", name); }

  hid_t     ds = H5Dget_space(dset);
  int const ndims = H5Sget_simple_extent_ndims(ds);
  return ndims;
}

template <int ND>
Eigen::DSizes<Index, ND> get_dims(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor {}", name); }

  hid_t     ds = H5Dget_space(dset);
  int const ndims = H5Sget_simple_extent_ndims(ds);
  if (ndims != ND) { Log::Fail("Tensor {}: Requested rank {}, on-disk rank {}", name, ND, ndims); }
  std::array<hsize_t, ND> hdims;
  H5Sget_simple_extent_dims(ds, hdims.data(), NULL);
  Eigen::DSizes<Index, ND> dims;
  std::reverse_copy(hdims.begin(), hdims.end(), dims.begin());
  return dims;
}

template <typename Scalar, int ND>
void load_tensor(Handle const &parent, std::string const &name, Eigen::Tensor<Scalar, ND> &tensor)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) { Log::Fail("Tensor {}: Requested rank {}, on-disk rank {}", name, ND, rank); }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND; ii++) {
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        "Tensor {}: expected dimensions were {}, but were {} on disk",
        name,
        fmt::join(tensor.dimensions(), ","),
        fmt::join(dims, ","));
    }
  }
  herr_t ret_value = H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail("Error reading tensor tensor {}, code: {}", name, ret_value);
  } else {
    Log::Print<Log::Level::High>("Read dataset: {}", name);
  }
}

template <typename Scalar, int CD>
void load_tensor_slab(Handle const &parent, std::string const &name, Index const index, Eigen::Tensor<Scalar, CD> &tensor)
{
  int const ND = CD + 1;
  hid_t     dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) { Log::Fail("Tensor {}: has rank {}, expected {}", name, rank, ND); }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND - 1; ii++) {   // Last dimension is SUPPOSED to be different
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        "Tensor {}: expected dimensions were {}, but were {} on disk",
        name,
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
  auto status = H5Sselect_hyperslab(ds, H5S_SELECT_SET, h5_start.data(), h5_stride.data(), h5_count.data(), h5_block.data());

  std::array<hsize_t, ND> mem_dims, mem_start, mem_stride, mem_count, mem_block;
  std::copy_n(dims.begin() + 1, CD, mem_dims.begin());
  std::fill_n(mem_start.begin(), CD, 0);
  std::fill_n(mem_stride.begin(), CD, 1);
  std::fill_n(mem_count.begin(), CD, 1);
  std::copy_n(dims.begin() + 1, CD, mem_block.begin());
  auto const mem_ds = H5Screate_simple(CD, mem_dims.data(), NULL);
  status = H5Sselect_hyperslab(mem_ds, H5S_SELECT_SET, mem_start.data(), mem_stride.data(), mem_count.data(), mem_block.data());

  status = H5Dread(dset, type<Scalar>(), mem_ds, ds, H5P_DEFAULT, tensor.data());
  if (status < 0) {
    Log::Fail("Tensor {}: Error reading slab {}. HD5 Message: {}", name, index, GetError());
  } else {
    Log::Print<Log::Level::High>("Read slab {} from tensor {}", index, name);
  }
}

template <typename Scalar, int ND>
Eigen::Tensor<Scalar, ND> load_tensor(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) { Log::Fail("Tensor {}: has rank {}, expected {}", name, rank, ND); }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  typename Eigen::Tensor<Scalar, ND>::Dimensions tDims;
  std::copy_n(dims.begin(), ND, tDims.begin());
  std::reverse(tDims.begin(), tDims.end()); // HD5=row-major, Eigen=col-major
  Eigen::Tensor<Scalar, ND> tensor(tDims);
  herr_t                    ret_value = H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail("Error reading tensor {}, code: {}", name, ret_value);
  } else {
    Log::Print<Log::Level::High>("Read tensor {}", name);
  }
  return tensor;
}

template <typename Derived>
Derived load_matrix(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open matrix '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank > 2) { Log::Fail("Matrix {}: has rank {} on disk, must be 1 or 2", name, rank); }

  std::array<hsize_t, 2> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  Derived matrix((rank == 2) ? dims[1] : dims[0], (rank == 2) ? dims[0] : 1);
  herr_t  ret_value = H5Dread(dset, type<typename Derived::Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, matrix.data());
  if (ret_value < 0) {
    Log::Fail("Error reading matrix {}, code: {}", name, ret_value);
  } else {
    Log::Print<Log::Level::High>("Read matrix {}", name);
  }
  return matrix;
}

Reader::Reader(std::string const &fname)
{
  if (!std::filesystem::exists(fname)) { Log::Fail("File does not exist: {}", fname); }
  Init();
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (handle_ < 0) { Log::Fail("Failed to open {}", fname); }
  Log::Print("Opened file to read: {}", fname);
  Log::Print<Log::Level::High>("Handle: {}", handle_);
}

Reader::~Reader()
{
  H5Fclose(handle_);
  Log::Print<Log::Level::High>("Closed handle: {}", handle_);
}

auto Reader::list() const -> std::vector<std::string> { return List(handle_); }

auto Reader::rank(std::string const &label) const -> Index { return getRank(handle_, label); }

template <int Rank>
auto Reader::dimensions(std::string const &label) const -> Eigen::DSizes<Index, Rank>
{
  return get_dims<Rank>(handle_, label);
}

template auto Reader::dimensions<1>(std::string const &) const -> Eigen::DSizes<Index, 1>;
template auto Reader::dimensions<2>(std::string const &) const -> Eigen::DSizes<Index, 2>;
template auto Reader::dimensions<3>(std::string const &) const -> Eigen::DSizes<Index, 3>;
template auto Reader::dimensions<4>(std::string const &) const -> Eigen::DSizes<Index, 4>;
template auto Reader::dimensions<5>(std::string const &) const -> Eigen::DSizes<Index, 5>;
template auto Reader::dimensions<6>(std::string const &) const -> Eigen::DSizes<Index, 6>;

template <typename T>
auto Reader::readTensor(std::string const &label) const -> T
{
  return load_tensor<typename T::Scalar, T::NumDimensions>(handle_, label);
}

template auto Reader::readTensor<I1>(std::string const &) const -> I1;
template auto Reader::readTensor<Re1>(std::string const &) const -> Re1;
template auto Reader::readTensor<Re2>(std::string const &) const -> Re2;
template auto Reader::readTensor<Re3>(std::string const &) const -> Re3;
template auto Reader::readTensor<Cx3>(std::string const &) const -> Cx3;
template auto Reader::readTensor<Cx4>(std::string const &) const -> Cx4;
template auto Reader::readTensor<Cx5>(std::string const &) const -> Cx5;
template auto Reader::readTensor<Cx6>(std::string const &) const -> Cx6;

template <typename T>
auto Reader::readSlab(std::string const &label, Index const ind) const -> T
{
  constexpr Index ND = T::NumDimensions;
  T               result(FirstN<ND>(dimensions<ND + 1>(label)));
  load_tensor_slab(handle_, label, ind, result);
  return result;
}

template auto Reader::readSlab<Cx3>(std::string const &, Index const) const -> Cx3;
template auto Reader::readSlab<Cx4>(std::string const &, Index const) const -> Cx4;

template <typename Derived>
auto Reader::readMatrix(std::string const &label) const -> Derived
{
  return load_matrix<Derived>(handle_, label);
}

template auto Reader::readMatrix<Eigen::MatrixXf>(std::string const &) const -> Eigen::MatrixXf;
template auto Reader::readMatrix<Eigen::MatrixXcf>(std::string const &) const -> Eigen::MatrixXcf;

template auto Reader::readMatrix<Eigen::ArrayXf>(std::string const &) const -> Eigen::ArrayXf;
template auto Reader::readMatrix<Eigen::ArrayXXf>(std::string const &) const -> Eigen::ArrayXXf;

auto Reader::readInfo() const -> Info
{
  // First get the Info struct
  hid_t const info_id = InfoType();
  hid_t const dset = H5Dopen(handle_, Keys::Info.c_str(), H5P_DEFAULT);
  CheckInfoType(dset);
  hid_t const space = H5Dget_space(dset);
  Info        info;
  CheckedCall(H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info), "Could not read info struct");
  CheckedCall(H5Dclose(dset), "Could not close info dataset");
  return info;
}

auto Reader::exists(std::string const &label) const -> bool { return Exists(handle_, label); }

auto Reader::readMeta() const -> std::map<std::string, float>
{
  auto meta_group = H5Gopen(handle_, Keys::Meta.c_str(), H5P_DEFAULT);
  if (meta_group < 0) {
    Log::Print<Log::Level::High>("No meta-data found in file handle {}", handle_);
    return {};
  }
  auto const                   names = List(meta_group);
  std::map<std::string, float> meta;
  herr_t                       status = 0;
  for (auto const &name : names) {
    hid_t const dset = H5Dopen(meta_group, name.c_str(), H5P_DEFAULT);
    float       value;
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    status = H5Dclose(dset);
    meta[name] = value;
  }
  status = H5Gclose(meta_group);
  if (status != 0) { Log::Fail("Could not load meta-data, code: {}", status); }
  return meta;
}

} // namespace HD5
} // namespace rl
