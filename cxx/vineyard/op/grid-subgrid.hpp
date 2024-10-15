#pragma once

#include "types.hpp"
#include <mutex>

namespace rl {

template <int ND, bool hasVCC, bool isVCC>
void GridToSubgrid(Eigen::Array<int16_t, ND, 1> const sg, CxNCMap<ND + 2 + hasVCC> const &x, CxN<ND + 2> &sx);
template <int ND, bool hasVCC, bool isVCC>
void SubgridToGrid(std::vector<std::mutex>           &mutexes,
                   Eigen::Array<int16_t, ND, 1> const sg,
                   CxNCMap<ND + 2> const             &sx,
                   CxNMap<ND + 2 + hasVCC>           &x);

} // namespace rl
