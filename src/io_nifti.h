#pragma once

#include "log.h"
#include "radial.h"

template <typename T, int ND>
extern void WriteNifti(
    RadialInfo const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname, Log &log);
template <typename T>
extern void WriteNifti(Eigen::Matrix<T, -1, -1> const &m, std::string const &fname, Log &log);