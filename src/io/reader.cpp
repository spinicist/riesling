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

auto Reader::dimensions(std::string const &label) const -> std::vector<Index>
{
  hid_t dset = H5Dopen(handle_, label.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor {}", label); }

  hid_t     ds = H5Dget_space(dset);
  int const ND = H5Sget_simple_extent_ndims(ds);
  std::vector<hsize_t> hdims(ND);
  H5Sget_simple_extent_dims(ds, hdims.data(), NULL);
  std::vector<Index> dims(ND);
  for (int ii = 0; ii < ND; ii++) {
    dims[ii] = hdims[ii];
  }
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  return dims;
}

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
auto Reader::readSlab(std::string const &label, std::vector<Index> const &sliceDims, std::vector<Index> const &sliceInds) const -> T
{
  if (sliceInds.size() != sliceDims.size()) {
    Log::Fail("Slice indices had size {}, slice dimensions had size {}", sliceInds.size(), sliceDims.size());
  }

  constexpr Index SliceRank = T::NumDimensions;

  hid_t     dset = H5Dopen(handle_, label.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor '{}'", label); }
  hid_t      ds = H5Dget_space(dset);
  auto const DiskRank = H5Sget_simple_extent_ndims(ds);

  if (SliceRank + sliceDims.size() != DiskRank) {
    Log::Fail("Requested {}D slice from {}D tensor with {} slicing dimensions", SliceRank, DiskRank, sliceDims.size());
  }

  std::vector<hsize_t> diskShape(DiskRank);
  H5Sget_simple_extent_dims(ds, diskShape.data(), NULL);
  std::reverse(diskShape.begin(), diskShape.end()); // HD5=row-major, Eigen=col-major
  for (int ii = 0; ii < sliceDims.size(); ii++) {
    if (diskShape[sliceDims[ii]] <= sliceInds[ii]) {
      Log::Fail("Tensor dimension {} has size {}, requested slice at index {}", sliceDims[ii], diskShape[sliceDims[ii]], sliceInds[ii]);
    }
  }
  std::vector<hsize_t> diskStart(DiskRank), diskStride(DiskRank), diskCount(DiskRank), diskBlock(DiskRank);
  std::fill_n(diskStart.begin(), DiskRank, 0);
  std::fill_n(diskStride.begin(), DiskRank, 1);
  std::fill_n(diskCount.begin(), DiskRank, 1);
  std::copy_n(diskShape.begin(), DiskRank, diskBlock.begin());
  for (int ii = 0; ii < sliceDims.size(); ii++) {
    diskStart[sliceDims[ii]] = sliceInds[ii];
    diskBlock[sliceDims[ii]] = 1;
  }
  // Figure out the non-slice dimensions
  std::vector<hsize_t> dims(SliceRank);
  int id = 0;
  for (int ii = 0; ii < DiskRank; ii++) {
    if (std::find(sliceDims.begin(), sliceDims.end(), ii) == std::end(sliceDims)) {
      dims[id++] = ii;
    }
  }

  std::vector<hsize_t> memShape(SliceRank), memStart(SliceRank), memStride(SliceRank), memCount(SliceRank);
  for (int ii = 0; ii < SliceRank; ii++) {
    memShape[ii] = diskShape[dims[ii]];
  }
  std::fill_n(memStart.begin(), SliceRank, 0);
  std::fill_n(memStride.begin(), SliceRank, 1);
  std::fill_n(memCount.begin(), SliceRank, 1);

   // Reverse back
  std::reverse(diskShape.begin(), diskShape.end());
  std::reverse(diskStart.begin(), diskStart.end());
  std::reverse(diskStride.begin(), diskStride.end());
  std::reverse(diskCount.begin(), diskCount.end());
  std::reverse(diskBlock.begin(), diskBlock.end());
  std::reverse(memShape.begin(), memShape.end());

  auto const mem_ds = H5Screate_simple(SliceRank, memShape.data(), NULL);
  auto status = H5Sselect_hyperslab(ds, H5S_SELECT_SET, diskStart.data(), diskStride.data(), diskCount.data(), diskBlock.data());
  status = H5Sselect_hyperslab(mem_ds, H5S_SELECT_SET, memStart.data(), memStride.data(), memCount.data(), memShape.data());

  T tensor;
  tensor.resize(memShape);
  status = H5Dread(dset, type<typename T::Scalar>(), mem_ds, ds, H5P_DEFAULT, tensor.data());
  if (status < 0) {
    Log::Fail("Tensor {}: Error reading slab. HD5 Message: {}", label, GetError());
  } else {
    Log::Print<Log::Level::High>("Read slab from tensor {}", label);
  }
  return tensor;
}

template auto Reader::readSlab<Cx2>(std::string const &, std::vector<Index> const &, std::vector<Index> const &) const -> Cx2;
template auto Reader::readSlab<Cx3>(std::string const &, std::vector<Index> const &, std::vector<Index> const &) const -> Cx3;
template auto Reader::readSlab<Cx4>(std::string const &, std::vector<Index> const &, std::vector<Index> const &) const -> Cx4;

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
