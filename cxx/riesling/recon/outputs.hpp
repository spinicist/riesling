#pragma once

#include "info.hpp"
#include "op/top.hpp"
#include "types.hpp"

namespace rl {
void WriteOutput(std::string const &fname, Cx5 const &img, Info const &info, std::string const &log = "");
void WriteResidual(
  std::string const &fname, Cx5 &noncart, Cx5Map &x, Info const &info, TOps::TOp<Cx, 5, 5>::Ptr A, Ops::Op<Cx>::Ptr M);
} // namespace rl