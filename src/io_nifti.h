#pragma once

#include "info.h"
#include "log.h"

template <typename T, int ND>
extern void WriteNifti(
    Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname, Log const &log);
