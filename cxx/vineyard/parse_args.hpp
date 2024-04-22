#pragma once

#include <args.hxx>
#include <map>
#include <optional>
#include <vector>

#include "trajectory.hpp"
#include "types.hpp"

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

void ParseCommand(args::Subparser &parser);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname);

struct Array2fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Array2f &x);
};

struct Array3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Array3f &x);
};

struct Vector3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Vector3f &x);
};

template <typename T>
struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<T> &x);
};

template <int N>
struct SzReader
{
  void operator()(std::string const &name, std::string const &value, rl::Sz<N> &x);
};

struct CoreOpts
{
  CoreOpts(args::Subparser &parser);
  args::Positional<std::string>                  iname, oname;
  args::ValueFlag<std::string>                   basisFile, residual, scaling;
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov;
  args::Flag                                     ndft;
};

struct PrecondOpts
{
  PrecondOpts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<float>       bias;
};

struct LsqOpts
{
  LsqOpts(args::Subparser &parser);
  args::ValueFlag<Index> its;
  args::ValueFlag<float> atol;
  args::ValueFlag<float> btol;
  args::ValueFlag<float> ctol;
  args::ValueFlag<float> λ;
};

void WriteOutput(std::string const &fname, rl::Cx5 const &img, rl::Info const &info, std::string const &log = "");
