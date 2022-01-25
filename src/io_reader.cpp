#include "io_reader.h"
#include "io_hd5.h"
#include "io_hd5.hpp"

#include <filesystem>

namespace HD5 {

Reader::Reader(std::string const &fname)
{
  if (!std::filesystem::exists(fname)) {
    Log::Fail(fmt::format("File does not exist: {}", fname));
  }
  Init();
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  Log::Print(FMT_STRING("Opened file {} for reading"), fname);
  Log::Debug(FMT_STRING("File handle: {}"), handle_);
}

Reader::~Reader()
{
  H5Fclose(handle_);
  Log::Debug(FMT_STRING("Closed HDF5 handle {}"), handle_);
}

template <typename T>
T Reader::readTensor(std::string const &label)
{
  return HD5::load_tensor<typename T::Scalar, T::NumDimensions>(handle_, label);
}

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

namespace {
herr_t AddName(hid_t id, const char *name, const H5L_info_t *linfo, void *opdata)
{
  auto names = reinterpret_cast<std::vector<std::string> *>(opdata);
  names->push_back(name);
  return 0;
}
} // namespace

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
    Log::Fail("Could not load info struct, code: {}", status);
  }

  R3 points(3, info.read_points, info.spokes);
  HD5::load_tensor(handle_, Keys::Trajectory, points);
  if (HD5::Exists(handle_, "echoes")) {
    I1 echoes(info.spokes);
    HD5::load_tensor(handle_, "echoes", echoes);
    Log::Debug("Read echoes successfully");
    traj_ = Trajectory(info, points, echoes);
  } else {
    Log::Debug("No echoes information in file");
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
  std::vector<std::string> names;
  H5Literate(meta_group, H5_INDEX_NAME, H5_ITER_INC, NULL, AddName, &names);

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
    Log::Fail("Could not load meta-data, code: {}", status);
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
    Log::Print("Using cached non-cartesion volume {}", index);
  } else {
    Log::Print("Reading non-cartesian volume {}", index);
    if (nc_.size() == 0) {
      nc_.resize(Sz3{traj_.info().channels, traj_.info().read_points, traj_.info().spokes});
    }
    HD5::load_tensor_slab(handle_, Keys::Noncartesian, index, nc_);
    currentNCVol_ = index;
  }
  return nc_;
}

} // namespace HD5
