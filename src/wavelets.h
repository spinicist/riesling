#pragma once

#include "log.h"
#include "types.h"

struct Wavelets
{
  Wavelets(long const N, long const levels, Log &io);

  std::tuple<Sz3, Sz3> pad_setup(Sz3 const &dims) const;
  void pad(Cx3 const &src, Cx3 &dest);
  void unpad(Cx3 const &src, Cx3 &dest);

  void encode(Cx3 &image);
  void decode(Cx3 &image);

private:
  void encode_dim(Cx3 &image, long const dim, long const level);
  void decode_dim(Cx3 &image, long const dim, long const level);
  long const N_, L_;
  Log &log_;
  R1 D_; // Coefficients
};
