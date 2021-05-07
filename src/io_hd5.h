#pragma once

#include "info.h"
#include "log.h"

#include <map>
#include <string>

namespace HD5 {

void Init();

struct Writer
{
  Writer(std::string const &fname, Log &log_);
  ~Writer();
  void writeInfo(Info const &info);
  void writeMeta(std::map<std::string, float> const &meta);
  void writeTrajectory(R3 const &traj);
  void writeNoncartesian(Cx4 const &allVolumes);
  void writeCartesian(Cx4 const &volume);
  void writeImage(R4 const &volumes);
  void writeImage(Cx4 const &volumes);
  void writeSDC(R2 const &sdc);

private:
  Log &log_;
  int64_t handle_;
};

struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname, Log &log);
  ~Reader();
  Info const &info() const;
  std::map<std::string, float> readMeta() const;
  R3 readTrajectory();
  R2 readSDC();
  void readNoncartesian(Cx4 &allVolumes);
  void readNoncartesian(long const index, Cx3 &volume);
  void readCartesian(Cx4 &volume);

private:
  Log &log_;
  int64_t handle_;
  Info info_;
};
} // namespace HD5
