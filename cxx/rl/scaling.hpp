#pragma once

#include "op/recon.hpp"

namespace rl {

auto ScaleImages(std::string const &type, Cx5 const &b) -> float;
auto ScaleData(std::string const &type, Ops::Op::Ptr const A, Ops::Op::Ptr const P, Ops::Op::Map const b) -> float;

} // namespace rl
