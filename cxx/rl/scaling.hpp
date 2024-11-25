#pragma once

#include "op/recon.hpp"

namespace rl {

auto ScaleData(std::string const &type, Ops::Op<Cx>::Ptr const A, Ops::Op<Cx>::Ptr const P, Ops::Op<Cx>::Map b) -> float;
void UnscaleData(float const scale, Ops::Op<Cx>::Vector &b);

} // namespace rl
