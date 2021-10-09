#include "io_hd5.h"
#include "io_hd5.hpp"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

namespace Keys {
std::string const Info = "info";
std::string const Meta = "meta";
std::string const Noncartesian = "noncartesian";
std::string const Cartesian = "cartesian";
std::string const Image = "image";
std::string const Trajectory = "trajectory";
std::string const Basis = "basis";
std::string const BasisImages = "basis-images";
std::string const Dynamics = "dynamics";
std::string const SDC = "sdc";
std::string const SENSE = "sense";
} // namespace Keys

namespace HD5 {

void Init(Log &log)
{
  static bool NeedsInit = true;

  if (NeedsInit) {

    auto err = H5open();
    // herr_t (*old_func)(long long, void *);
    // void *old_client_data;
    // hid_t errorStack;
    // err = H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
    err = H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
    if (err < 0) {
      Log::Fail("Could not initialise HDF5, code: {}", err);
    }
    NeedsInit = false;
  }
}

Writer::Writer(std::string const &fname, Log &log)
    : log_(log)
{
  Init(log_);
  handle_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail(FMT_STRING("Could not open file {} for writing"), fname);
  } else {
    log_.info(FMT_STRING("Opened file {} for writing data"), fname);
  }
}

Writer::~Writer()
{
  H5Fclose(handle_);
}

hid_t InfoType(Log &log)
{
  hid_t info_id = H5Tcreate(H5T_COMPOUND, sizeof(Info));
  hsize_t sz3[1] = {3};
  hid_t long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, sz3);
  hid_t float3_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz3);
  hsize_t sz9[1] = {9};
  hid_t float9_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz9);
  herr_t status;
  status = H5Tinsert(info_id, "matrix", HOFFSET(Info, matrix), long3_id);
  status = H5Tinsert(info_id, "voxel_size", HOFFSET(Info, voxel_size), float3_id);
  status = H5Tinsert(info_id, "read_points", HOFFSET(Info, read_points), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "read_gap", HOFFSET(Info, read_gap), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_hi", HOFFSET(Info, spokes_hi), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_lo", HOFFSET(Info, spokes_lo), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "lo_scale", HOFFSET(Info, lo_scale), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "channels", HOFFSET(Info, channels), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "type", HOFFSET(Info, type), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "volumes", HOFFSET(Info, volumes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "echoes", HOFFSET(Info, echoes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "tr", HOFFSET(Info, tr), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "origin", HOFFSET(Info, origin), float3_id);
  status = H5Tinsert(info_id, "direction", HOFFSET(Info, direction), float9_id);
  if (status) {
    Log::Fail("Could not create Info struct type in HDF5, code: {}", status);
  }
  return info_id;
}

void CheckInfoType(hid_t handle, Log &log)
{
  // Hard code for now until the fields in InfoType are replaced with some kind of auto-gen
  constexpr int N = 14;
  std::array<std::string, N> const names{"matrix",
                                         "voxel_size",
                                         "read_points",
                                         "read_gap",
                                         "spokes_hi",
                                         "spokes_lo",
                                         "lo_scale",
                                         "channels",
                                         "type",
                                         "volumes",
                                         "echoes",
                                         "tr",
                                         "origin",
                                         "direction"};

  auto const dtype = H5Dget_type(handle);
  int n_members = H5Tget_nmembers(dtype);
  if (n_members != N) {
    Log::Fail("Header info had {} members, should be {}", n_members, N);
  }
  // Re-orderd fields are okay. Missing is not
  for (auto const &check_name : names) {
    bool found = false;
    for (int ii = 0; ii < N; ii++) {
      std::string const member_name(H5Tget_member_name(dtype, ii));
      if (member_name == check_name) {
        found = true;
        break;
      }
    }
    if (!found) {
      Log::Fail("Field {} not found in header info", check_name);
    }
  }
}

void Writer::writeInfo(Info const &info)
{
  log_.info("Writing info struct");
  hid_t info_id = InfoType(log_);
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  hid_t const dset =
      H5Dcreate(handle_, Keys::Info.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail("Could not create info struct, code: {}", dset);
  }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) {
    Log::Fail("Could not write Info struct, code: {}", status);
  }
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  log_.info("Writing meta data");
  auto m_group = H5Gcreate(handle_, Keys::Meta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t status;
  for (auto const &kvp : meta) {
    hid_t const dset = H5Dcreate(
        m_group, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  if (status != 0) {
    throw std::runtime_error("Exception occured storing meta-data");
  }
}

void Writer::writeTrajectory(Trajectory const &t)
{
  writeInfo(t.info());
  HD5::store_tensor(handle_, Keys::Trajectory, t.points(), log_);
}

void Writer::writeSDC(R2 const &sdc)
{
  HD5::store_tensor(handle_, Keys::SDC, sdc, log_);
}

void Writer::writeSENSE(Cx4 const &s)
{
  HD5::store_tensor(handle_, Keys::SENSE, s, log_);
}

void Writer::writeNoncartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Noncartesian, t, log_);
}

void Writer::writeCartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Cartesian, t, log_);
}

void Writer::writeImage(R4 const &t)
{
  HD5::store_tensor(handle_, Keys::Image, t, log_);
}

void Writer::writeImage(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Image, t, log_);
}

void Writer::writeBasis(R2 const &b)
{
  HD5::store_tensor(handle_, Keys::Basis, b, log_);
}

void Writer::writeDynamics(R2 const &d)
{
  HD5::store_tensor(handle_, Keys::Dynamics, d, log_);
}

void Writer::writeRealMatrix(R2 const &m, std::string const &k)
{
  HD5::store_tensor(handle_, k, m, log_);
}

void Writer::writeReal4(R4 const &d, std::string const &k)
{
  HD5::store_tensor(handle_, k, d, log_);
}

void Writer::writeReal5(R5 const &d, std::string const &k)
{
  HD5::store_tensor(handle_, k, d, log_);
}

void Writer::writeBasisImages(Cx5 const &bi)
{
  HD5::store_tensor(handle_, Keys::BasisImages, bi, log_);
}

Reader::Reader(std::string const &fname, Log &log)
    : log_{log}
{
  if (!std::filesystem::exists(fname)) {
    Log::Fail(fmt::format("File does not exist: {}", fname));
  }
  Init(log_);
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  log_.info(FMT_STRING("Opened file {} for reading"), fname);
}

Reader::~Reader()
{
  H5Fclose(handle_);
}

herr_t AddName(hid_t id, const char *name, const H5L_info_t *linfo, void *opdata)
{
  auto names = reinterpret_cast<std::vector<std::string> *>(opdata);
  names->push_back(name);
  return 0;
}

std::map<std::string, float> Reader::readMeta() const
{
  auto meta_group = H5Gopen(handle_, Keys::Meta.c_str(), H5P_DEFAULT);
  if (meta_group < 0) {
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

Info Reader::readInfo()
{
  log_.info("Reading info");
  hid_t const info_id = InfoType(log_);
  hid_t const dset = H5Dopen(handle_, Keys::Info.c_str(), H5P_DEFAULT);
  CheckInfoType(dset, log_);
  hid_t const space = H5Dget_space(dset);
  Info info;
  herr_t status = H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info);
  status = H5Dclose(dset);
  if (status != 0) {
    Log::Fail("Could not load info struct, code: {}", status);
  }
  return info;
}

Trajectory Reader::readTrajectory()
{
  Info info = readInfo();
  log_.info("Reading trajectory");
  R3 points(3, info.read_points, info.spokes_total());
  HD5::load_tensor(handle_, Keys::Trajectory, points, log_);
  return Trajectory(info, points, log_);
}

R2 Reader::readSDC(Info const &info)
{
  log_.info("Reading SDC");
  R2 sdc(info.read_points, info.spokes_total());
  HD5::load_tensor(handle_, Keys::SDC, sdc, log_);
  return sdc;
}

void Reader::readNoncartesian(Cx4 &all)
{
  log_.info("Reading all non-cartesian data");
  HD5::load_tensor(handle_, Keys::Noncartesian, all, log_);
}

void Reader::readNoncartesian(long const index, Cx3 &vol)
{
  log_.info("Reading non-cartesian volume {}", index);
  HD5::load_tensor_slab(handle_, Keys::Noncartesian, index, vol, log_);
}

void Reader::readCartesian(Cx4 &grid)
{
  log_.info("Reading cartesian data");
  HD5::load_tensor(handle_, Keys::Cartesian, grid, log_);
}

void Reader::readSENSE(Cx4 &s)
{
  log_.info("Reading SENSE maps");
  HD5::load_tensor(handle_, Keys::SENSE, s, log_);
}

Cx4 Reader::readSENSE()
{
  log_.info("Reading SENSE maps");
  auto const dims = HD5::get_dims<4>(handle_, Keys::SENSE, log_);
  Cx4 sense(dims);
  HD5::load_tensor(handle_, Keys::SENSE, sense, log_);
  return sense;
}

R2 Reader::readBasis()
{
  log_.info("Reading basis");
  return HD5::load_tensor<float, 2>(handle_, Keys::Basis, log_);
}

R2 Reader::readRealMatrix(std::string const &k)
{
  log_.info("Reading {}", k);
  return HD5::load_tensor<float, 2>(handle_, k, log_);
}

Cx5 Reader::readBasisImages()
{
  log_.info("Reading basis");
  return HD5::load_tensor<Cx, 5>(handle_, Keys::BasisImages, log_);
}

} // namespace HD5
