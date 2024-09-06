#include "io/reader.hpp"

#include "log.hpp"
#include <filesystem>
#include <hdf5.h>
#include <hdf5_hl.h>

namespace rl {
namespace HD5 {

Reader::Reader(std::string const &fname, bool const altX)
  : owner_{true}
  , altComplex_{altX}
{
  if (!std::filesystem::exists(fname)) { Log::Fail("File does not exist: {}", fname); }
  Init();
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (handle_ < 0) { Log::Fail("Failed to open {}", fname); }
  Log::Print("Opened file to read: {}", fname);
  Log::Debug("Handle: {}", handle_);
}

Reader::Reader(Handle const fid, bool const altX)
  : handle_{fid}
  , owner_{false}
  , altComplex_{altX}
{
  Init();
  Log::Print("Reading from HDF5 id {}", fid);
}

Reader::~Reader()
{
  if (owner_) {
    H5Fclose(handle_);
    Log::Debug("Closed handle: {}", handle_);
  }
}

auto Reader::list() const -> std::vector<std::string> { return List(handle_); }

auto Reader::order(std::string const &name) const -> Index
{
  hid_t dset = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor {}", name); }
  hid_t     ds = H5Dget_space(dset);
  int const ndims = H5Sget_simple_extent_ndims(ds);
  CheckedCall(H5Dclose(dset), "Could not close dataset");
  return ndims;
}

auto Reader::dimensions(std::string const &label) const -> std::vector<Index>
{
  hid_t dset = H5Dopen(handle_, label.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor {}", label); }

  hid_t                ds = H5Dget_space(dset);
  int const            ND = H5Sget_simple_extent_ndims(ds);
  std::vector<hsize_t> hdims(ND);
  H5Sget_simple_extent_dims(ds, hdims.data(), NULL);
  std::vector<Index> dims(ND);
  for (int ii = 0; ii < ND; ii++) {
    dims[ii] = hdims[ii];
  }
  std::reverse(dims.begin(), dims.end()); // HD5=row-major, Eigen=col-major
  return dims;
}

auto Reader::listNames(std::string const &name) const -> std::vector<std::string>
{
  hid_t ds = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
  if (ds < 0) { Log::Fail("Could not open tensor '{}'", name); }
  hid_t                    dspace = H5Dget_space(ds);
  int const                ndims = H5Sget_simple_extent_ndims(dspace);
  std::vector<std::string> names(ndims);
  char                     buffer[64] = {0};
  for (Index ii = 0; ii < ndims; ii++) {
    H5DSget_label(ds, ii, buffer, sizeof(buffer));
    names[ii] = std::string(buffer);
  }
  std::reverse(names.begin(), names.end());
  CheckedCall(H5Dclose(ds), "Could not close dataset");
  return names;
}

template <typename T> auto Reader::readTensor(std::string const &name) const -> T
{
  constexpr auto ND = T::NumDimensions;
  using Scalar = typename T::Scalar;
  hid_t dset = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
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
    Log::Debug("Read tensor {}, shape {}", name, tDims);
  }
  return tensor;
}

template auto Reader::readTensor<I1>(std::string const &) const -> I1;
template auto Reader::readTensor<Re1>(std::string const &) const -> Re1;
template auto Reader::readTensor<Re2>(std::string const &) const -> Re2;
template auto Reader::readTensor<Re3>(std::string const &) const -> Re3;
template auto Reader::readTensor<Cx2>(std::string const &) const -> Cx2;
template auto Reader::readTensor<Cx3>(std::string const &) const -> Cx3;
template auto Reader::readTensor<Cx4>(std::string const &) const -> Cx4;
template auto Reader::readTensor<Cx5>(std::string const &) const -> Cx5;
template auto Reader::readTensor<Cx6>(std::string const &) const -> Cx6;

template <int N> auto Reader::dimensionNames(std::string const &name) const -> DimensionNames<N>
{
  if (N != order(name)) { Log::Fail("Asked for {} dimension names, but {} order tensor", N, order(name)); }
  hid_t ds = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
  if (ds < 0) { Log::Fail("Could not open tensor '{}'", name); }
  DimensionNames<N> names;
  for (Index ii = 0; ii < N; ii++) {
    char buffer[64] = {0};
    H5DSget_label(ds, ii, buffer, sizeof(buffer));
    names[ii] = std::string(buffer);
  }
  std::reverse(names.begin(), names.end());
  H5Dclose(ds);
  return names;
}

template auto Reader::dimensionNames<3>(std::string const &) const -> DimensionNames<3>;
template auto Reader::dimensionNames<4>(std::string const &) const -> DimensionNames<4>;
template auto Reader::dimensionNames<5>(std::string const &) const -> DimensionNames<5>;
template auto Reader::dimensionNames<6>(std::string const &) const -> DimensionNames<6>;

template <typename T> auto Reader::readSlab(std::string const &label, std::vector<IndexPair> const &chips) const -> T
{
  constexpr Index SlabOrder = T::NumDimensions;

  hid_t dset = H5Dopen(handle_, label.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open tensor '{}'", label); }
  hid_t       ds = H5Dget_space(dset);
  Index const DiskOrder = H5Sget_simple_extent_ndims(ds);

  if (SlabOrder + (Index)chips.size() != DiskOrder) {
    Log::Fail("Requested {}D slice from {}D tensor with {} chips", SlabOrder, DiskOrder, chips.size());
  }

  std::vector<hsize_t> diskShape(DiskOrder);
  H5Sget_simple_extent_dims(ds, diskShape.data(), NULL);
  std::reverse(diskShape.begin(), diskShape.end()); // HD5=row-major, Eigen=col-major
  for (size_t ii = 0; ii < chips.size(); ii++) {
    auto const chip = chips[ii];
    if (DiskOrder <= chip.dim) { Log::Fail("Tensor {} has order {} requested chip dim {}", label, DiskOrder, chip.dim); }
    if (diskShape[chip.dim] <= (hsize_t)chip.index) {
      Log::Fail("Tensor {} dim {} has size {} requested index {}", label, chip.dim, diskShape[chip.dim], chip.index);
    }
  }
  std::vector<hsize_t> diskStart(DiskOrder), diskStride(DiskOrder), diskCount(DiskOrder), diskBlock(DiskOrder);
  std::fill_n(diskStart.begin(), DiskOrder, 0);
  std::fill_n(diskStride.begin(), DiskOrder, 1);
  std::fill_n(diskCount.begin(), DiskOrder, 1);
  std::copy_n(diskShape.begin(), DiskOrder, diskBlock.begin());
  for (size_t ii = 0; ii < chips.size(); ii++) {
    auto const chip = chips[ii];
    diskStart[chip.dim] = chip.index;
    diskBlock[chip.dim] = 1;
  }
  // Figure out the non-chip dimensions
  std::vector<hsize_t> dims(SlabOrder);
  int                  id = 0;
  for (int ii = 0; ii < DiskOrder; ii++) {
    if (std::find_if(chips.begin(), chips.end(), [ii](IndexPair const c) { return c.dim == ii; }) == std::end(chips)) {
      dims[id++] = ii;
    }
  }

  std::vector<hsize_t> memShape(SlabOrder), memStart(SlabOrder), memStride(SlabOrder), memCount(SlabOrder);
  for (int ii = 0; ii < SlabOrder; ii++) {
    memShape[ii] = diskShape[dims[ii]];
  }
  std::fill_n(memStart.begin(), SlabOrder, 0);
  std::fill_n(memStride.begin(), SlabOrder, 1);
  std::fill_n(memCount.begin(), SlabOrder, 1);

  T tensor;
  tensor.resize(memShape);

  // Reverse back to HDF5 order
  std::reverse(diskShape.begin(), diskShape.end());
  std::reverse(diskStart.begin(), diskStart.end());
  std::reverse(diskStride.begin(), diskStride.end());
  std::reverse(diskCount.begin(), diskCount.end());
  std::reverse(diskBlock.begin(), diskBlock.end());
  std::reverse(memShape.begin(), memShape.end());

  auto const mem_ds = H5Screate_simple(SlabOrder, memShape.data(), NULL);
  auto       status =
    H5Sselect_hyperslab(ds, H5S_SELECT_SET, diskStart.data(), diskStride.data(), diskCount.data(), diskBlock.data());
  status = H5Sselect_hyperslab(mem_ds, H5S_SELECT_SET, memStart.data(), memStride.data(), memCount.data(), memShape.data());
  status = H5Dread(dset, type<typename T::Scalar>(), mem_ds, ds, H5P_DEFAULT, tensor.data());
  if (status < 0) {
    Log::Fail("Tensor {}: Error reading slab. HD5 Message: {}", label, GetError());
  } else {
    Log::Debug("Read slab from tensor {}", label);
  }
  return tensor;
}

template auto Reader::readSlab<Cx2>(std::string const &, std::vector<IndexPair> const &) const -> Cx2;
template auto Reader::readSlab<Cx3>(std::string const &, std::vector<IndexPair> const &) const -> Cx3;
template auto Reader::readSlab<Cx4>(std::string const &, std::vector<IndexPair> const &) const -> Cx4;

template <typename Derived> auto Reader::readMatrix(std::string const &name) const -> Derived
{
  hid_t dset = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open matrix '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank > 2) { Log::Fail("Matrix {}: has rank {} on disk, must be 1 or 2", name, rank); }

  std::array<hsize_t, 2> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  Derived matrix((rank == 2) ? dims[1] : dims[0], (rank == 2) ? dims[0] : 1);
  herr_t  ret_value =
    H5Dread(dset, type<typename Derived::Scalar>(altComplex_), ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, matrix.data());
  if (ret_value < 0) {
    Log::Fail("Error reading matrix {}, code: {}", name, ret_value);
  } else {
    Log::Debug("Read matrix {}", name);
  }
  return matrix;
}

template auto Reader::readMatrix<Eigen::MatrixXf>(std::string const &) const -> Eigen::MatrixXf;
template auto Reader::readMatrix<Eigen::MatrixXd>(std::string const &) const -> Eigen::MatrixXd;
template auto Reader::readMatrix<Eigen::MatrixXcf>(std::string const &) const -> Eigen::MatrixXcf;
template auto Reader::readMatrix<Eigen::MatrixXcd>(std::string const &) const -> Eigen::MatrixXcd;

template auto Reader::readMatrix<Eigen::ArrayXf>(std::string const &) const -> Eigen::ArrayXf;
template auto Reader::readMatrix<Eigen::ArrayXXf>(std::string const &) const -> Eigen::ArrayXXf;

auto Reader::readString(std::string const &name) const -> std::string {
  hid_t dset = H5Dopen(handle_, name.c_str(), H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not open dataset '{}'", name); }
  hid_t      ds = H5Dget_space(dset);
  auto const rank = H5Sget_simple_extent_ndims(ds);
  if (rank != 1) { Log::Fail("String {}: has rank {} on disk, must be 1", name, rank); }
  std::array<hsize_t, 1> dims;
  H5Sget_simple_extent_dims(ds, dims.data(), NULL);
  hid_t const tid = H5Tcopy(H5T_C_S1);
  H5Tset_size(tid, H5T_VARIABLE);
  H5Tset_cset(tid, H5T_CSET_UTF8);
  char *rdata[dims[0]];
  CheckedCall(H5Dread(dset, tid, ds, H5S_ALL, H5P_DATASET_XFER_DEFAULT, rdata), "Could not read string");
  CheckedCall(H5Dclose(dset), "Could not close string dataset");
  Log::Debug("Read string {}", name);
  return std::string(rdata[0]);
}

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
auto Reader::exists(std::string const &dset, std::string const &attr) const -> bool
{
  return H5Aexists_by_name(handle_, dset.c_str(), attr.c_str(), H5P_DEFAULT);
}

auto Reader::readMeta() const -> std::map<std::string, float>
{
  auto meta_group = H5Gopen(handle_, Keys::Meta.c_str(), H5P_DEFAULT);
  if (meta_group < 0) {
    Log::Debug("No meta-data found in file handle {}", handle_);
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

auto Reader::readAttributeFloat(std::string const &dset, std::string const &attr) const -> float
{
  float      val;
  auto const attrH = H5Aopen_by_name(handle_, dset.c_str(), attr.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  CheckedCall(H5Aread(attrH, H5T_NATIVE_FLOAT, &val), fmt::format("reading attribute {} from {}", attr, dset));
  return val;
}

auto Reader::readAttributeInt(std::string const &dset, std::string const &attr) const -> long
{
  long       val;
  auto const attrH = H5Aopen_by_name(handle_, dset.c_str(), attr.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  CheckedCall(H5Aread(attrH, H5T_NATIVE_LONG, &val), fmt::format("reading attribute {} from {}", attr, dset));
  return val;
}

template <int N> auto Reader::readAttributeSz(std::string const &dset, std::string const &attr) const -> Sz<N>
{
  hsize_t szN[1] = {N};
  hid_t   long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, szN);
  Sz<N>   val;

  auto const attrH = H5Aopen_by_name(handle_, dset.c_str(), attr.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  CheckedCall(H5Aread(attrH, long3_id, val.data()), fmt::format("reading attribute {} from {}", attr, dset));
  return val;
}

template auto Reader::readAttributeSz<1>(std::string const &, std::string const &attr) const -> Sz<1>;
template auto Reader::readAttributeSz<2>(std::string const &, std::string const &attr) const -> Sz<2>;
template auto Reader::readAttributeSz<3>(std::string const &, std::string const &attr) const -> Sz<3>;

} // namespace HD5
} // namespace rl
