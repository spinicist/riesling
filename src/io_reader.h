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
  Reader(std::string const &fname);
  ~Reader();
  std::map<std::string, float> readMeta() const;
  Info readInfo();
  Trajectory readTrajectory();
  R2 readSDC(Info const &info);

  void readCartesian(Cx5 &volume);
  Cx3 const &noncartesian(Index const index); // This will be cached
  Cx4 readSENSE();
  R2 readBasis();
  R2 readRealMatrix(std::string const &label);
  Cx5 readBasisImages();

  template <typename T>
  T readTensor(std::string const &label);
  template <typename T>
  void readTensor(std::string const &label, T &tensor); // Read with size checks
private:
    int64_t handle_;
  Index currentNCVol_;
  Cx3 nc_;
};

} // namespace HD5
