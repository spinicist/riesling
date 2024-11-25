#pragma once

#include "../types.hpp"
#include <mutex>

namespace rl {

template <int ND>
void GridToSubgrid(
  Eigen::Array<int16_t, ND, 1> const sg, bool const hasVCC, bool const isVCC, CxNCMap<ND + 2> const &x, CxN<ND + 2> &sx);
template <int ND>
void SubgridToGrid(std::vector<std::mutex>           &mutexes,
                   Eigen::Array<int16_t, ND, 1> const sg,
                   bool const                         hasVCC,
                   bool const                         isVCC,
                   CxNCMap<ND + 2> const             &sx,
                   CxNMap<ND + 2>                    &x);

} // namespace rl
