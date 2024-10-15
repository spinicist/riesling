#pragma once

#include "info.hpp"
#include "io/hd5-core.hpp"
#include "op/top.hpp"
#include "types.hpp"

namespace rl {
template <int ND>
void WriteOutput(
  std::string const &cmd, std::string const &fname, CxN<ND> const &img, HD5::DimensionNames<ND> const &dims, Info const &info);

void WriteResidual(std::string const                &cmd,
                   std::string const                &fname,
                   Cx5                              &noncart,
                   Cx5Map const                     &x,
                   Info const                       &info,
                   typename TOps::TOp<Cx, 5, 5>::Ptr A,
                   Ops::Op<Cx>::Ptr                  M,
                   HD5::DimensionNames<5> const     &dims);
} // namespace rl