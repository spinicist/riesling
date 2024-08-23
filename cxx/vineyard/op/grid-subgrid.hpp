#pragma once

#include "types.hpp"

namespace rl {

template <int ND, bool hasVCC, bool isVCC> void GridToSubgrid(Sz<ND> const sg, CxNCMap<ND + 2 + hasVCC> const &x, CxN<ND + 2> &sx);
template <int ND, bool hasVCC, bool isVCC> void SubgridToGrid(Sz<ND> const sg, CxNCMap<ND + 2> const &sx, CxNMap<ND + 2 + hasVCC> &x);

} // namespace rl
