#pragma once

#include "log.h"
#include "types.h"

namespace rl {

struct Wavelets
{
  Wavelets(Index const N, Index const levels);

  std::tuple<Sz3, Sz3> pad_setup(Sz3 const &dims) const;
  void pad(Cx3 const &src, Cx3 &dest);
  void unpad(Cx3 const &src, Cx3 &dest);

  void encode(Cx3 &image);
  void decode(Cx3 &image);

private:
  void encode_dim(Cx3 &image, Index const dim, Index const level);
  void decode_dim(Cx3 &image, Index const dim, Index const level);
  Index const N_, L_;

  R1 D_; // Coefficients
};
} // namespace rl
