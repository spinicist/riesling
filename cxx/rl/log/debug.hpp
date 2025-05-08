#pragma once

#include "log.hpp"

#include "../io/hd5-core.hpp"
#include "../types.hpp"

namespace rl {
namespace Log {

void SetDebugFile(std::string const &fname);
auto IsDebugging() -> bool;
void EndDebugging();

template <typename Scalar, size_t ND>
void Tensor(std::string const &name, HD5::Shape<ND> const &shape, Scalar const *data, HD5::DNames<ND> const &dims);

} // namespace Log
} // namespace rl
