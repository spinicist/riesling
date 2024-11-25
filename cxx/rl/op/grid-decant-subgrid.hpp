#pragma once

#include "../types.hpp"
#include <mutex>

namespace rl {

template <int ND>
void GridToDecant(Eigen::Array<int16_t, ND, 1> const sg, CxN<ND + 2> const &sk, CxNCMap<ND + 1> const &x, CxN<ND + 2> &sx);
template <int ND>
void DecantToGrid(std::vector<std::mutex>           &mutexes,
                  Eigen::Array<int16_t, ND, 1> const sg,
                  CxN<ND + 2> const                 &sk,
                  CxNCMap<ND + 2> const             &sx,
                  CxNMap<ND + 1>                    &x);

} // namespace rl
