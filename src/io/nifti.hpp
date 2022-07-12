#pragma once

#include "info.h"
#include "log.h"

namespace rl {
template <typename T, int ND>
extern void WriteNifti(
    rl::Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname);
}
