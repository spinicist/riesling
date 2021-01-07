#include "io_hd5.h"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

HD5Writer::HD5Writer(std::string const &fname, Log &log)
    : log_(log)
{
  HD5::init();
  handle_ = HD5::open_file(fname, HD5::Mode::WriteOnly);
  data_ = HD5::create_group(handle_, "data");
  log_.info(FMT_STRING("Opened file {} for writing data"), fname);
}

HD5Writer::~HD5Writer()
{
  HD5::close_file(handle_);
}

void HD5Writer::writeData(long const index, Cx3 const &data)
{
  auto const label = fmt::format("{:04d}", index);
  log_.info(FMT_STRING("Writing volume {}"), label);
  HD5::store_tensor(data_, label, data);
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
  data_ = HD5::open_group(handle_, "data");
  HD5::load_info(handle_, info_);
  log_.info(FMT_STRING("Opened file {} for reading"), fname);
}

HD5Reader::~HD5Reader()
{
  HD5::close_group(data_);
  HD5::close_file(handle_);
}

Info const &HD5Reader::info() const
{
  return info_;
}

void HD5Reader::readData(long const index, Cx3 &ks)
{
  assert(ks.dimension(0) == info_.channels);
  assert(ks.dimension(1) == info_.read_points);
  assert(ks.dimension(2) == info_.spokes_total());
  assert(index < info_.volumes);

  auto const label = fmt::format("{:04d}", index);
  log_.info(FMT_STRING("Reading volume {}"), index);
  HD5::load_tensor(data_, label, ks);
}

void HD5Reader::readData(Cx4 &ks)
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
