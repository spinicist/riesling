#include "io_hd5.h"
#include "io_hd5.hpp"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

namespace HD5 {

std::string const KeyInfo = "info";
std::string const KeyMeta = "meta";
std::string const KeyNoncartesian = "noncartesian";
std::string const KeyCartesian = "cartesian";
std::string const KeyImage = "image";
std::string const KeyTrajectory = "trajectory";
std::string const KeySDC = "sdc";

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
      log.fail("Could not initialise HDF5, code: {}", err);
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
    log_.fail(FMT_STRING("Could not open file {} for writing"), fname);
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
  status = H5Tinsert(info_id, "tr", HOFFSET(Info, tr), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "origin", HOFFSET(Info, origin), float3_id);
  status = H5Tinsert(info_id, "direction", HOFFSET(Info, direction), float9_id);
  if (status) {
    log.fail("Could not create Info struct type in HDF5, code: {}", status);
  }
  return info_id;
}

void Writer::writeInfo(Info const &info)
{
  log_.info("Writing info struct");
  hid_t info_id = InfoType(log_);
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  hid_t const dset =
      H5Dcreate(handle_, KeyInfo.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) {
    log_.fail("Could not create info struct, code: {}", dset);
  }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) {
    log_.fail("Could not write Info struct, code: {}", status);
  }
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  log_.info("Writing meta data");
  auto m_group = H5Gcreate(handle_, KeyMeta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

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
  HD5::store_tensor(handle_, KeyTrajectory, t.points(), log_);
}

void Writer::writeSDC(R2 const &sdc)
{
  HD5::store_tensor(handle_, KeySDC, sdc, log_);
}

void Writer::writeNoncartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, KeyNoncartesian, t, log_);
}

void Writer::writeCartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, KeyCartesian, t, log_);
}

void Writer::writeImage(R4 const &t)
{
  HD5::store_tensor(handle_, KeyImage, t, log_);
}

void Writer::writeImage(Cx4 const &t)
{
  HD5::store_tensor(handle_, KeyImage, t, log_);
}

Reader::Reader(std::string const &fname, Log &log)
    : log_{log}
{
  if (!std::filesystem::exists(fname)) {
    log_.fail(fmt::format("File does not exist: {}", fname));
  }
  Init(log_);
  handle_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  log_.info(FMT_STRING("Opened file {} for reading"), fname);

  hid_t const info_id = InfoType(log_);
  hid_t const dset = H5Dopen(handle_, KeyInfo.c_str(), H5P_DEFAULT);
  hid_t const space = H5Dget_space(dset);
  herr_t status = H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info_);
  status = H5Dclose(dset);
  if (status != 0) {
    log_.fail("Could not load info struct, code: {}", status);
  }
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
  auto meta_group = H5Gopen(handle_, KeyMeta.c_str(), H5P_DEFAULT);
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
    log_.fail("Could not load meta-data, code: {}", status);
  }
  return meta;
}

Info const &Reader::info() const
{
  return info_;
}

Trajectory Reader::readTrajectory()
{
  log_.info("Reading trajectory");
  R3 points(3, info_.read_points, info_.spokes_total());
  HD5::load_tensor(handle_, KeyTrajectory, points, log_);
  return Trajectory(info_, points, log_);
}

R2 Reader::readSDC()
{
  log_.info("Reading SDC");
  R2 sdc(info_.read_points, info_.spokes_total());
  HD5::load_tensor(handle_, KeySDC, sdc, log_);
  return sdc;
}

void Reader::readNoncartesian(Cx4 &all)
{
  log_.info("Reading all non-cartesian data");
  HD5::load_tensor(handle_, KeyNoncartesian, all, log_);
}

void Reader::readNoncartesian(long const index, Cx3 &vol)
{
  log_.info("Reading non-cartesian volume {}", index);
  HD5::load_tensor_slab(handle_, KeyNoncartesian, index, vol, log_);
}

void Reader::readCartesian(Cx4 &grid)
{
  log_.info("Reading cartesian data");
  HD5::load_tensor(handle_, KeyCartesian, grid, log_);
}

} // namespace HD5
