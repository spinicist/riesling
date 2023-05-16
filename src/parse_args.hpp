#pragma once

#include <args.hxx>
#include <vector>
#include <optional>

#include "types.hpp"
#include "trajectory.hpp"

extern args::Group global_group;
extern args::HelpFlag help;
extern args::Flag verbose;

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname);
void ParseCommand(args::Subparser &parser);

struct Vector3fReader
{
  void operator()(std::string const &name, std::string const &value, Eigen::Vector3f &x);
};

template <typename T>
struct VectorReader
{
  void operator()(std::string const &name, std::string const &value, std::vector<T> &x);
};

struct Sz2Reader
{
  void operator()(std::string const &name, std::string const &value, rl::Sz2 &x);
};

struct Sz3Reader
{
  void operator()(std::string const &name, std::string const &value, rl::Sz3 &x);
};

auto ReadBasis(std::string const &basisFile) -> std::optional<rl::Re2>;

// Helper function to generate a good output name
std::string OutName(
  std::string const &iName, std::string const &oName, std::string const &suffix, std::string const &extension = "h5");

struct CoreOpts
{
  CoreOpts(args::Subparser &parser);
  args::Positional<std::string> iname;
  args::ValueFlag<std::string> oname, basisFile, ktype, scaling;
  args::ValueFlag<float> osamp, fov;
  args::ValueFlag<Index> bucketSize;
  args::Flag residImage, residKSpace, keepTrajectory;
};

void WriteOutput(
  CoreOpts &opts,
  rl::Cx5 const &img,
  std::string const &suffix,
  rl::Trajectory const &traj,
  std::string const &log,
  rl::Cx5 const &residImage = rl::Cx5(),
  rl::Cx5 const &residKSpace = rl::Cx5(),
  std::map<std::string, float> const &meta = std::map<std::string, float>());

