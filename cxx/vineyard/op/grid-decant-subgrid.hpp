#pragma once

#include "types.hpp"
#include <mutex>

namespace rl {

template <int ND> void GridToDecant(Sz<ND> const sg, CxN<ND + 2> const &sk, CxNCMap<ND + 1> const &x, CxN<ND + 2> &sx);
template <int ND>
void DecantToGrid(
  std::vector<std::mutex> &mutexes, Sz<ND> const sg, CxN<ND + 2> const &sk, CxNCMap<ND + 2> const &sx, CxNMap<ND + 1> &x);

} // namespace rl
