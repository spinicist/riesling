#pragma once

#include "hd5.h"
#include "info.h"
#include "log.h"

#include <map>
#include <string>

struct HD5Writer
{
  HD5Writer(std::string const &fname, Log &log_);
  ~HD5Writer();
  void writeVolumes(Cx4 const &data);
  void writeVolume(long const index, Cx3 const &data);
  void writeData(Cx4 const &data, std::string const &label);
  void writeTrajectory(R3 const &traj);
  void writeInfo(Info const &info);
  void writeMeta(std::map<std::string, float> const &meta);

private:
  Log &log_;
  HD5::Handle handle_;
};

struct HD5Reader
{
  HD5Reader(HD5Reader const &) = delete;
  HD5Reader(std::string const &fname, Log &log);
  ~HD5Reader();
  Info const &info() const;
  void readVolume(long const index, Cx3 &data);
  void readVolumes(Cx4 &data);
  void readData(Cx4 &data, std::string const &label);
  R3 readTrajectory();
  std::map<std::string, float> readMeta() const;

private:
  Log &log_;
  HD5::Handle handle_;
  Info info_;
};
