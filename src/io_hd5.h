#pragma once

#include "log.h"
#include "trajectory.h"

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
  void writeTrajectory(Trajectory const &traj);
  void writeNoncartesian(Cx4 const &allVolumes);
  void writeCartesian(Cx4 const &volume);
  void writeImage(R4 const &volumes);
  void writeImage(Cx4 const &volumes);
  void writeSDC(R2 const &sdc);
  void writeSENSE(Cx4 const &sense);
  void writeBasis(R2 const &basis);
  void writeDynamics(R2 const &dynamics);
  void writeRealMatrix(R2 const &mat, std::string const &label);
  void writeReal4(R4 const &d, std::string const &label);
  void writeReal5(R5 const &d, std::string const &label);
  void writeBasisImages(Cx5 const &basis);

private:
  Log &log_;
  int64_t handle_;
};

struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname, Log &log);
  ~Reader();
  std::map<std::string, float> readMeta() const;
  Info readInfo();
  Trajectory readTrajectory();
  R2 readSDC(Info const &info);
  void readNoncartesian(Cx4 &allVolumes);
  void readNoncartesian(long const index, Cx3 &volume);
  void readCartesian(Cx4 &volume);
  void readSENSE(Cx4 &sense);
  Cx4 readSENSE();
  R2 readBasis();
  R2 readRealMatrix(std::string const &label);
  Cx5 readBasisImages();

private:
  Log &log_;
  int64_t handle_;
};
} // namespace HD5
