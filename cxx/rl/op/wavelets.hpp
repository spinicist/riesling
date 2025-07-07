#pragma once

#include "top.hpp"

namespace rl::TOps {

template<int ND>
struct Wavelets final : TOp<ND, ND>
{
  TOP_INHERIT(ND, ND)

  Wavelets(Sz<ND> const shape, Index const N, std::vector<Index> const dims);

  TOP_DECLARE(Wavelets)

  static auto PaddedShape(Sz<ND> const shape, std::vector<Index> const dims) -> Sz<ND>;

private:
  void  dimLoops(InMap x, bool const rev) const;
  void  wav1(Index const N, bool const rev, Cx1 &x) const;
  Index N_;
  Re1   Cc_, Cr_; // Coefficients
  std::vector<Index> dims_;
};

} // namespace rl::TOps
