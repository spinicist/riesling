#include "io_reader.h"
#include "io_hd5.h"
#include "io_hd5.hpp"

#include <filesystem>

namespace HD5 {

Reader::Reader(std::string const &fname, Log &log)
  : log_{log}
  , currentNCVol_{-1}
{
  if (!std::filesystem::exists(fname)) {
    Log::Fail(fmt::format("File does not exist: {}", fname));
  }
  Init();
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

Cx3 const &Reader::noncartesian(long const index)
{
  if (index == currentNCVol_) {
    log_.info("Using cached non-cartesion volume {}", index);
  } else {
    log_.info("Reading non-cartesian volume {}", index);
    if (nc_.size() == 0) {
      auto const dims4 = HD5::get_dims<4>(handle_, Keys::Noncartesian, log_);
      Cx3::Dimensions dims3;
      dims3[0] = dims4[0];
      dims3[1] = dims4[1];
      dims3[2] = dims4[2];
      nc_.resize(dims3);
    }
    HD5::load_tensor_slab(handle_, Keys::Noncartesian, index, nc_, log_);
    currentNCVol_ = index;
  }
  return nc_;
}

void Reader::readCartesian(Cx4 &grid)
{
  log_.info("Reading cartesian data");
  HD5::load_tensor(handle_, Keys::Cartesian, grid, log_);
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
