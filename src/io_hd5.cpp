#include "io_hd5.h"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

RadialWriter::RadialWriter(std::string const &fname, Log &log)
    : log_(log)
{
  HD5::init();
  handle_ = HD5::open_file(fname, HD5::Mode::WriteOnly);
  data_ = HD5::create_group(handle_, "data");
  log_.info(FMT_STRING("Opened file {} for writing data"), fname);
}

RadialWriter::~RadialWriter()
{
  HD5::close_file(handle_);
}

void RadialWriter::writeData(long const index, Cx3 const &radial)
{
  auto const label = fmt::format("{:04d}", index);
  log_.info(FMT_STRING("Writing volume {}"), label);
  HD5::store_tensor(data_, label, radial);
}

void RadialWriter::writeTrajectory(R3 const &traj)
{
  log_.info("Writing trajectory");
  HD5::store_tensor(handle_, "traj", traj);
}

void RadialWriter::writeInfo(RadialInfo const &info)
{
  log_.info("Writing radial header");
  HD5::store_info(handle_, info);
}

void RadialWriter::writeMeta(std::map<std::string, float> const &meta)
{
  log_.info("Writing meta data");
  auto m_group = HD5::create_group(handle_, "meta");
  HD5::store_map(m_group, meta);
}

RadialReader::RadialReader(std::string const &fname, Log &log)
    : log_{log}
{
  if (!std::filesystem::exists(fname)) {
    log_.fail(fmt::format("File does not exist: {}", fname));
  }
  HD5::init();
  handle_ = HD5::open_file(fname, HD5::Mode::ReadOnly);
  data_ = HD5::open_group(handle_, "data");
  HD5::load_info(handle_, info_);
  log_.info(FMT_STRING("Opened file {} for reading"), fname);
}

RadialReader::~RadialReader()
{
  HD5::close_group(data_);
  HD5::close_file(handle_);
}

RadialInfo const &RadialReader::info() const
{
  return info_;
}

void RadialReader::readData(long const index, Cx3 &ks)
{
  assert(ks.dimension(0) == info_.channels);
  assert(ks.dimension(1) == info_.read_points);
  assert(ks.dimension(2) == info_.spokes_total());
  assert(index < info_.volumes);

  auto const label = fmt::format("{:04d}", index);
  log_.info(FMT_STRING("Reading volume {}"), index);
  HD5::load_tensor(data_, label, ks);
}

void RadialReader::readData(Cx4 &ks)
{
  assert(ks.dimension(0) == info_.channels);
  assert(ks.dimension(1) == info_.read_points);
  assert(ks.dimension(2) == info_.spokes_total());
  assert(ks.dimension(3) == info_.volumes);

  Cx3 volume(ks.dimension(0), ks.dimension(1), ks.dimension(2));
  for (long ivol = 0; ivol < info_.volumes; ivol++) {
    auto const label = fmt::format("{:04d}", ivol);
    log_.info(FMT_STRING("Reading volume {}"), ivol);
    HD5::load_tensor(data_, label, volume);
    ks.chip(ivol, 3) = volume;
  }
}

R3 RadialReader::readTrajectory()
{
  log_.info("Reading trajectory");
  R3 trajectory(3, info_.read_points, info_.spokes_total());
  HD5::load_tensor(handle_, "traj", trajectory);
  return trajectory;
}

std::map<std::string, float> RadialReader::readMeta() const
{
  auto m_group = HD5::open_group(handle_, "meta");
  std::map<std::string, float> meta;
  HD5::load_map(m_group, meta);
  return meta;
}
