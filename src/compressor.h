#pragma once

#include "log.h"
#include "types.h"

struct Compressor
{
  Compressor(Cx3 const &ks, long const nc, Log &log);
  long out_channels() const;
  void compress(Cx3 const &source, Cx3 &dest);
  void compress(Cx4 const &source, Cx4 &dest);

private:
  Log &log_;
  Eigen::MatrixXcf psi_;
};