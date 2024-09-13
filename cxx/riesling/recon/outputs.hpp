#pragma once

#include "info.hpp"
#include "io/hd5-core.hpp"
#include "op/top.hpp"
#include "types.hpp"

namespace rl {
template <int ND>
void WriteOutput(std::string const             &cmd,
                 std::string const             &fname,
                 CxN<ND> const                 &img,
                 HD5::DimensionNames<ND> const &dims,
                 Info const                    &info,
                 std::string const             &log = "");
template <int ND>
void WriteResidual(std::string const                 &cmd,
                   std::string const                 &fname,
                   Cx5                               &noncart,
                   CxNCMap<ND> const                 &x,
                   Info const                        &info,
                   typename TOps::TOp<Cx, ND, 5>::Ptr A,
                   Ops::Op<Cx>::Ptr                   M,
                   HD5::DimensionNames<ND> const     &dims);
} // namespace rl