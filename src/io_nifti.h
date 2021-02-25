#pragma once

#include "info.h"
#include "log.h"

template <typename T, int ND>
extern void WriteNifti(
    Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname, Log const &log);
template <typename T>
extern void WriteNifti(Eigen::Matrix<T, -1, -1> const &m, std::string const &fname, Log const &log);