#pragma once

#include "io_hd5.h"
#include "log.h"
#include "trajectory.h"

#include <map>
#include <string>

namespace HD5 {

struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname, Log &log);
  ~Reader();
  std::map<std::string, float> readMeta() const;
  Info readInfo();
  Trajectory readTrajectory();
  R2 readSDC(Info const &info);

  void readCartesian(Cx4 &volume);
  Cx3 const &noncartesian(long const index); // This will be cached
  Cx4 readSENSE();
  R2 readBasis();
  R2 readRealMatrix(std::string const &label);
  Cx5 readBasisImages();

private:
  Log &log_;
  int64_t handle_;
  long currentNCVol_;
  Cx3 nc_;
};

} // namespace HD5
