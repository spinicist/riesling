#pragma once

#include "hd5.h"
#include "log.h"
#include "radial.h"

#include <map>
#include <string>

struct RadialWriter
{
  RadialWriter(std::string const &fname, Log &log_);
  ~RadialWriter();
  void writeData(long const index, Cx3 const &radial);
  void writeTrajectory(R3 const &traj);
  void writeInfo(RadialInfo const &info);
  void writeMeta(std::map<std::string, float> const &meta);

private:
  Log &log_;
  HD5::Handle handle_, data_;
};

struct RadialReader
{
  RadialReader(RadialReader const &) = delete;
  RadialReader(std::string const &fname, Log &log);
  ~RadialReader();
  RadialInfo const &info() const;
  void readData(long const index, Cx3 &radial);
  void readData(Cx4 &ks);
  R3 readTrajectory();
  std::map<std::string, float> readMeta() const;

private:
  Log &log_;
  HD5::Handle handle_, data_;
  RadialInfo info_;
};
