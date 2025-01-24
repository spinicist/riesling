#pragma once

#include "op/recon.hpp"

namespace rl {

auto ScaleImages(std::string const &type, Cx5 const &b) -> float;
auto ScaleData(std::string const &type, Ops::Op<Cx>::Ptr const A, Ops::Op<Cx>::Ptr const P, Ops::Op<Cx>::Map const b) -> float;

} // namespace rl
