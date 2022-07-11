#pragma once

#include <args.hxx>
#include <vector>

#include "types.h"
#include "trajectory.h"

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
  void operator()(std::string const &name, std::string const &value, Sz2 &x);
};

struct Sz3Reader
{
  void operator()(std::string const &name, std::string const &value, Sz3 &x);
};

// Helper function to generate a good output name
std::string OutName(
  std::string const &iName, std::string const &oName, std::string const &suffix, std::string const &extension = "h5");

void WriteOutput(
  Cx5 const &img,
  std::string const &iname,
  std::string const &oname,
  std::string const &suffix,
  bool const keepTrajectory,
  Trajectory const &traj);

// Helper function for getting a good volume to take SENSE maps from
Index ValOrLast(Index const val, Index const last);

struct CoreOpts
{
  CoreOpts(args::Subparser &parser);
  args::Positional<std::string> iname;
  args::ValueFlag<std::string> oname, ktype;
  args::ValueFlag<float> osamp;
  args::ValueFlag<Index> bucketSize;
  args::ValueFlag<std::string> basisFile;
  args::Flag keepTrajectory;
};

struct ExtraOpts
{
  ExtraOpts(args::Subparser &parser);
  args::ValueFlag<float> iter_fov, out_fov;
};
