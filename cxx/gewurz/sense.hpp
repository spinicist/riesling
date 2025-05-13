#pragma once

#include "rl/io/hd5-core.hpp"
#include <complex>

namespace gw {

auto GetSENSE(std::string const &path, rl::HD5::Shape<3> const mat) -> std::vector<std::complex<float>>;

}