#pragma once

#include "types.hpp"

#include <args.hxx>

struct Array2fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Array2f &x);
};

struct Array3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Array3f &x);
};

using Array3fFlag = args::ValueFlag<Eigen::Array3f, Array3fReader>;

struct ArrayXfReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::ArrayXf &x);
};

struct ArrayXiReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::ArrayXi &x);
};

struct Vector3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Vector3f &x);
};

template <typename T> struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<T> &x);
};

template <typename T> using VectorFlag = args::ValueFlag<std::vector<T>, VectorReader<T>>;

template <int N> struct SzReader
{
  void operator()(std::string const &name, std::string const &value, rl::Sz<N> &x);
};
