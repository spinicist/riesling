#pragma once

#include "log.hpp"

#include "../io/hd5-core.hpp"
#include "../types.hpp"

namespace rl {
namespace Log {

void SetDebugFile(std::string const &fname);
auto IsDebugging() -> bool;
void EndDebugging();

template <typename Scalar, int ND>
void Tensor(std::string const &name, Sz<ND> const &shape, Scalar const *data, HD5::DimensionNames<ND> const &dims);

} // namespace Log
} // namespace rl
