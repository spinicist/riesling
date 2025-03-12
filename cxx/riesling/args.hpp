#pragma once

#include "rl/types.hpp"

#include <args.hxx>

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

void SetLogging(std::string const &name);
void ParseCommand(args::Subparser &parser);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname);

template <int N> struct SzReader
{
  void operator()(std::string const &name, std::string const &value, rl::Sz<N> &x);
};

template <int ND> using SzFlag = args::ValueFlag<rl::Sz<ND>, SzReader<ND>>;

template <typename T, int ND> struct ArrayReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Array<T, ND, 1> &x);
};

template <typename T, int ND> using ArrayFlag = args::ValueFlag<Eigen::Array<T, ND, 1>, ArrayReader<T, ND>>;

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

struct Matrix3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Matrix3f &x);
};

template <typename T> struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<T> &x);
};

template <typename T> using VectorFlag = args::ValueFlag<std::vector<T>, VectorReader<T>>;
