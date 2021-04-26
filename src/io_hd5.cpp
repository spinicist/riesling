#include "io_hd5.h"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

HD5Writer::HD5Writer(std::string const &fname, Log &log)
    : log_(log)
{
  HD5::init();
  handle_ = HD5::open_file(fname, HD5::Mode::WriteOnly);
  log_.info(FMT_STRING("Opened file {} for writing data"), fname);
}

HD5Writer::~HD5Writer()
{
  HD5::close_file(handle_);
}

void HD5Writer::writeVolumes(Cx4 const &data)
{
  auto const hVols = HD5::create_group(handle_, "volumes");
  for (long ii = 0; ii < data.dimension(3); ii++) {
    Cx3 vol = data.chip(ii, 3);
    auto const label = fmt::format("{:04d}", index);
    log_.info(FMT_STRING("Writing volume {}"), label);
    HD5::store_tensor(hVols, label, vol);
  }
  HD5::close_group(hVols);
}

void HD5Writer::writeVolume(long const ivol, Cx3 const &data)
{
  auto const hVols =
      (ivol == 0) ? HD5::create_group(handle_, "volumes") : HD5::open_group(handle_, "volumes");

  auto const label = fmt::format("{:04d}", 0);
  log_.info(FMT_STRING("Writing volume {}"), label);
  HD5::store_tensor(hVols, label, data);
  HD5::close_group(hVols);
}

void HD5Writer::writeData(Cx4 const &data, std::string const &label)
{
  log_.info(FMT_STRING("Writing data '{}'"), label);
  HD5::store_tensor(handle_, label, data);
}

void HD5Writer::writeTrajectory(R3 const &traj)
{
  log_.info("Writing trajectory");
  HD5::store_tensor(handle_, "traj", traj);
}

void HD5Writer::writeInfo(Info const &info)
{
  log_.info("Writing info struct");
  HD5::store_info(handle_, info);
}

void HD5Writer::writeMeta(std::map<std::string, float> const &meta)
{
  log_.info("Writing meta data");
  auto m_group = HD5::create_group(handle_, "meta");
  HD5::store_map(m_group, meta);
}

HD5Reader::HD5Reader(std::string const &fname, Log &log)
    : log_{log}
{
  if (!std::filesystem::exists(fname)) {
    log_.fail(fmt::format("File does not exist: {}", fname));
  }
  HD5::init();
  handle_ = HD5::open_file(fname, HD5::Mode::ReadOnly);
  HD5::load_info(handle_, info_);
  log_.info(FMT_STRING("Opened file {} for reading"), fname);
}

HD5Reader::~HD5Reader()
{
  HD5::close_file(handle_);
}

Info const &HD5Reader::info() const
{
  return info_;
}

void HD5Reader::readVolume(long const index, Cx3 &ks)
{
  auto const hVol = HD5::open_group(handle_, "volumes");
  assert(ks.dimension(0) == info_.channels);
  assert(ks.dimension(1) == info_.read_points);
  assert(ks.dimension(2) == info_.spokes_total());
  assert(index < info_.volumes);

  auto const label = fmt::format("{:04d}", index);
  log_.info(FMT_STRING("Reading volume {}"), index);
  HD5::load_tensor(hVol, label, ks);
  HD5::close_group(hVol);
}

void HD5Reader::readVolumes(Cx4 &ks)
{
  assert(ks.dimension(0) == info_.channels);
  assert(ks.dimension(1) == info_.read_points);
  assert(ks.dimension(2) == info_.spokes_total());
  assert(ks.dimension(3) == info_.volumes);

  auto const hVol = HD5::open_group(handle_, "volumes");
  Cx3 volume(ks.dimension(0), ks.dimension(1), ks.dimension(2));
  for (long ivol = 0; ivol < info_.volumes; ivol++) {
    auto const label = fmt::format("{:04d}", ivol);
    log_.info(FMT_STRING("Reading volume {}"), ivol);
    HD5::load_tensor(hVol, label, volume);
    ks.chip(ivol, 3) = volume;
  }
  HD5::close_group(hVol);
}

void HD5Reader::readData(Cx4 &ks, std::string const &handle)
{
  log_.info(FMT_STRING("Reading data '{}'"), handle);
  HD5::load_tensor(handle_, "data", ks);
}

R3 HD5Reader::readTrajectory()
{
  log_.info("Reading trajectory");
  R3 trajectory(3, info_.read_points, info_.spokes_total());
  HD5::load_tensor(handle_, "traj", trajectory);
  return trajectory;
}

std::map<std::string, float> HD5Reader::readMeta() const
{
  auto m_group = HD5::open_group(handle_, "meta");
  std::map<std::string, float> meta;
  HD5::load_map(m_group, meta);
  return meta;
}
