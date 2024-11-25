#pragma once

#include "../info.hpp"
#include "../log.hpp"

namespace rl {
template <typename T, int ND>
extern void WriteNifti(rl::Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname);
}
