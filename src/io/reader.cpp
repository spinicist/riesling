#include "io/reader.hpp"

#include <filesystem>

namespace HD5 {

template <int ND>
Eigen::DSizes<Index, ND> get_dims(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor {}"), name);
  }

  hid_t ds = H5Dget_space(dset);
  int const ndims = H5Sget_simple_extent_ndims(ds);
  if (ndims != ND) {
    Log::Fail(FMT_STRING("In tensor {}, number of dimensions {} does match on-disk number {}"), name, ndims, ND);
  }
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
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor {}: has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND; ii++) {
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        FMT_STRING("Tensor {}: expected dimensions were {}, but were {} on disk"),
        name,
        fmt::join(tensor.dimensions(), ","),
        fmt::join(dims, ","));
    }
  }
  herr_t ret_value = H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail(FMT_STRING("Error reading tensor tensor {}, code: {}"), name, ret_value);
  } else {
    Log::Debug(FMT_STRING("Read dataset: {}"), name);
  }
}

template <typename Scalar, int CD>
void load_tensor_slab(
  Handle const &parent, std::string const &name, Index const index, Eigen::Tensor<Scalar, CD> &tensor)
{
  int const ND = CD + 1;
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor {}: has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < ND - 1; ii++) {   // Last dimension is SUPPOSED to be different
    if ((Index)dims[ii] != tensor.dimension(ii)) {
      Log::Fail(
        FMT_STRING("Tensor {}: expected dimensions were {}, but were {} on disk"),
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
  auto status =
    H5Sselect_hyperslab(ds, H5S_SELECT_SET, h5_start.data(), h5_stride.data(), h5_count.data(), h5_block.data());

  std::array<hsize_t, ND> mem_dims, mem_start, mem_stride, mem_count, mem_block;
  std::copy_n(dims.begin() + 1, CD, mem_dims.begin());
  std::fill_n(mem_start.begin(), CD, 0);
  std::fill_n(mem_stride.begin(), CD, 1);
  std::fill_n(mem_count.begin(), CD, 1);
  std::copy_n(dims.begin() + 1, CD, mem_block.begin());
  auto const mem_ds = H5Screate_simple(CD, mem_dims.data(), NULL);
  status = H5Sselect_hyperslab(
    mem_ds, H5S_SELECT_SET, mem_start.data(), mem_stride.data(), mem_count.data(), mem_block.data());

  status = H5Dread(dset, type<Scalar>(), mem_ds, ds, H5P_DEFAULT, tensor.data());
  if (status < 0) {
    Log::Fail(FMT_STRING("Tensor {}: Error reading slab {}. HD5 Message: {}"), name, index, HD5::GetError());
  } else {
    Log::Debug(FMT_STRING("Read slab {} from tensor {}"), index, name);
  }
}

template <typename Scalar, int ND>
Eigen::Tensor<Scalar, ND> load_tensor(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open tensor '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != ND) {
    Log::Fail(FMT_STRING("Tensor {}: has rank {}, expected {}"), name, rank, ND);
  }

  std::array<hsize_t, ND> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  typename Eigen::Tensor<Scalar, ND>::Dimensions tDims;
  std::copy_n(dims.begin(), ND, tDims.begin());
  std::reverse(tDims.begin(), tDims.end()); // HD5=row-major, Eigen=col-major
  Eigen::Tensor<Scalar, ND> tensor(tDims);
  herr_t ret_value = H5Dread(dset, type<Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, tensor.data());
  if (ret_value < 0) {
    Log::Fail(FMT_STRING("Error reading tensor {}, code: {}"), name, ret_value);
  } else {
    Log::Debug(FMT_STRING("Read tensor {}"), name);
  }
  return tensor;
}

template <typename Derived>
Derived load_matrix(Handle const &parent, std::string const &name)
{
  hid_t dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not open matrix '{}'"), name);
  }
  hid_t ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank > 2) {
    Log::Fail(FMT_STRING("Matrix {}: has rank {} on disk, must be 1 or 2"), name, rank);
  }

  std::array<hsize_t, 2> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  Derived matrix((rank == 2) ? dims[1] : dims[0], (rank == 2) ? dims[0] : 1);
  herr_t ret_value =
    H5Dread(dset, type<typename Derived::Scalar>(), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, matrix.data());
  if (ret_value < 0) {
    Log::Fail(FMT_STRING("Error reading matrix {}, code: {}"), name, ret_value);
  } else {
    Log::Debug(FMT_STRING("Read matrix {}"), name);
  }
  return matrix;
}

Reader::Reader(std::string const &fname)
{
  if (!std::filesystem::exists(fname)) {
    Log::Fail(FMT_STRING("File does not exist: {}"), fname);
  }
  Init();
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail(FMT_STRING("Failed to open {}"), fname);
  }
  Log::Print(FMT_STRING("Opened file to read: {}"), fname);
  Log::Debug(FMT_STRING("Handle: {}"), handle_);
}

Reader::~Reader()
{
  H5Fclose(handle_);
  Log::Debug(FMT_STRING("Closed handle: {}"), handle_);
}

std::vector<std::string> Reader::list()
{
  return HD5::List(handle_);
}

template <typename T>
T Reader::readTensor(std::string const &label)
{
  return HD5::load_tensor<typename T::Scalar, T::NumDimensions>(handle_, label);
}

template R1 Reader::readTensor<R1>(std::string const &);
template R2 Reader::readTensor<R2>(std::string const &);
template Cx3 Reader::readTensor<Cx3>(std::string const &);
template Cx4 Reader::readTensor<Cx4>(std::string const &);
template Cx5 Reader::readTensor<Cx5>(std::string const &);
template Cx6 Reader::readTensor<Cx6>(std::string const &);

template <typename T>
void Reader::readTensor(std::string const &label, T &tensor)
{
  HD5::load_tensor(handle_, label, tensor);
}

template void Reader::readTensor<Cx4>(std::string const &, Cx4 &);
template void Reader::readTensor<Cx5>(std::string const &, Cx5 &);
template void Reader::readTensor<Cx6>(std::string const &, Cx6 &);

template <typename Derived>
Derived Reader::readMatrix(std::string const &label)
{
  return HD5::load_matrix<Derived>(handle_, label);
}

template Eigen::MatrixXf Reader::readMatrix<Eigen::MatrixXf>(std::string const &);
template Eigen::MatrixXcf Reader::readMatrix<Eigen::MatrixXcf>(std::string const &);

template Eigen::ArrayXf Reader::readMatrix<Eigen::ArrayXf>(std::string const &);
template Eigen::ArrayXXf Reader::readMatrix<Eigen::ArrayXXf>(std::string const &);

void Check(std::string const &name, Index const dval, Index const ival)
{
  if (dval != ival) {
    Log::Fail(FMT_STRING("Number of {} in data {} does not match info {}"), name, dval, ival);
  }
}

RieslingReader::RieslingReader(std::string const &fname)
  : Reader(fname)
  , currentNCVol_{-1}
{
  // First get the Info struct
  hid_t const info_id = InfoType();
  hid_t const dset = H5Dopen(handle_, Keys::Info.c_str(), H5P_DEFAULT);
  CheckInfoType(dset);
  hid_t const space = H5Dget_space(dset);
  Info info;
  herr_t status = H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info);
  status = H5Dclose(dset);
  if (status != 0) {
    Log::Fail(FMT_STRING("Could not load info struct, code: {}"), status);
  }

  R3 points(3, info.read_points, info.spokes);
  HD5::load_tensor(handle_, Keys::Trajectory, points);
  if (HD5::Exists(handle_, "frames")) {
    I1 frames(info.spokes);
    HD5::load_tensor(handle_, "frames", frames);
    Log::Debug(FMT_STRING("Read frames successfully"));
    traj_ = Trajectory(info, points, frames);
  } else {
    Log::Debug(FMT_STRING("No frames information in file"));
    traj_ = Trajectory(info, points);
  }

  // For non-cartesian, check dimension sizes
  if (Exists(handle_, Keys::Noncartesian)) {
    auto const dims = HD5::get_dims<4>(handle_, Keys::Noncartesian);
    Check("channels", dims[0], info.channels);
    Check("read-points", dims[1], info.read_points);
    Check("spokes", dims[2], info.spokes);
    Check("volumes", dims[3], info.volumes);
  }
}

std::map<std::string, float> RieslingReader::readMeta() const
{
  auto meta_group = H5Gopen(handle_, Keys::Meta.c_str(), H5P_DEFAULT);
  if (meta_group < 0) {
    Log::Debug(FMT_STRING("No meta-data found in file handle {}"), handle_);
    return {};
  }
  auto const names = HD5::List(meta_group);
  std::map<std::string, float> meta;
  herr_t status = 0;
  for (auto const &name : names) {
    hid_t const dset = H5Dopen(meta_group, name.c_str(), H5P_DEFAULT);
    float value;
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    status = H5Dclose(dset);
    meta[name] = value;
  }
  status = H5Gclose(meta_group);
  if (status != 0) {
    Log::Fail(FMT_STRING("Could not load meta-data, code: {}"), status);
  }
  return meta;
}

Trajectory const &RieslingReader::trajectory() const
{
  return traj_;
}

Cx3 const &RieslingReader::noncartesian(Index const index)
{
  if (index == currentNCVol_) {
    Log::Print(FMT_STRING("Using cached non-cartesion volume {}"), index);
  } else {
    Log::Print(FMT_STRING("Reading non-cartesian volume {}"), index);
    if (nc_.size() == 0) {
      nc_.resize(Sz3{traj_.info().channels, traj_.info().read_points, traj_.info().spokes});
    }
    HD5::load_tensor_slab(handle_, Keys::Noncartesian, index, nc_);
    currentNCVol_ = index;
  }
  return nc_;
}

} // namespace HD5
