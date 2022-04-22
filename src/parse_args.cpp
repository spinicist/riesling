#include "parse_args.h"
#include "io/io.h"
#include "tensorOps.h"
#include "threads.h"
#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <fmt/format.h>
#include <scn/scn.h>

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None}, {1, Log::Level::Info}, {2, Log::Level::Progress}, {3, Log::Level::Debug}};
}

void Vector3fReader::operator()(
  std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  float x, y, z;
  auto result = scn::scan(value, "{},{},{}", x, y, z);
  if (!result) {
    Log::Fail(
      FMT_STRING("Could not read vector for {} from value {} because {}"),
      name,
      value,
      result.error());
  }
  v.x() = x;
  v.y() = y;
  v.z() = z;
}

template <typename T>
void VectorReader<T>::operator()(
  std::string const &name, std::string const &input, std::vector<T> &values)
{
  T val;
  auto result = scn::scan(input, "{}", val);
  if (result) {
    // Values will have been default initialized. Reset
    values.clear();
    values.push_back(val);
    while ((result = scn::scan(result.range(), ",{}", val))) {
      values.push_back(val);
    }
  } else {
    Log::Fail(FMT_STRING("Could not read argument for {}"), name);
  }
}

template struct VectorReader<float>;
template struct VectorReader<Index>;

void Sz2Reader::operator()(std::string const &name, std::string const &value, Sz2 &v)
{
  Index i, j;
  auto result = scn::scan(value, "{},{}", i, j);
  if (!result) {
    Log::Fail(FMT_STRING("Could not read {} from '{}': {}"), name, value, result.error());
  }
  v = Sz2{i, j};
}

void Sz3Reader::operator()(std::string const &name, std::string const &value, Sz3 &v)
{
  Index i, j, k;
  auto result = scn::scan(value, "{},{},{}", i, j, k);
  if (!result) {
    Log::Fail(FMT_STRING("Could not read {} from '{}': {}"), name, value, result.error());
  }
  v = Sz3{i, j, k};
}

args::Group global_group("GLOBAL OPTIONS");
args::HelpFlag help(global_group, "H", "Show this help message", {'h', "help"});
args::Flag verbose(global_group, "V", "Print logging messages to stdout", {'v', "verbose"});
args::MapFlag<int, Log::Level>
  verbosity(global_group, "V", "Talk more (values 0-3)", {"verbosity"}, levelMap);
args::ValueFlag<std::string> debug(global_group, "F", "Write debug images to file", {"debug"});
args::ValueFlag<Index> nthreads(global_group, "N", "Limit number of threads", {"nthreads"});

Log::Level GetVerbosity(bool const v)
{
  if (v) {
    return Log::Level::Info;
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    std::string env_level(env_p);
    if (env_level == "1") {
    return Log::Level::Info;
    } else if (env_level == "2") {
      return Log::Level::Progress;
    } else if (env_level == "3") {
      return Log::Level::Debug;
    } else {
      return Log::Level::None;
    }
  } else {
    return Log::Level::None;
  }
}

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  parser.Parse();
  Log::SetLevel(verbosity ? verbosity.Get() : GetVerbosity(verbose.Get()));
  Log::Print(FMT_STRING("Starting: {}"), parser.GetCommand().Name());
  if (debug) {
    Log::SetDebugFile(debug.Get());
  }
  if (!iname) {
    throw args::Error("No input file specified");
  }
  if (nthreads) {
    Log::Print(FMT_STRING("Using {} threads"), nthreads.Get());
    Threads::SetGlobalThreadCount(nthreads.Get());
  }
}

void ParseCommand(args::Subparser &parser)
{
  parser.Parse();
  Log::SetLevel(verbosity ? verbosity.Get() : (verbose ? Log::Level::Info : Log::Level::None));
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads.Get());
  }
  Log::Print(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());
}

std::string OutName(
  std::string const &iName,
  std::string const &oName,
  std::string const &suffix,
  std::string const &extension)
{
  return fmt::format(
    FMT_STRING("{}-{}.{}"),
    oName.empty() ? std::filesystem::path(iName).filename().replace_extension().string() : oName,
    suffix,
    extension);
}

Index ValOrLast(Index const val, Index const vols)
{
  if (val < 0) {
    return vols - 1;
  } else {
    return std::min(val, vols - 1);
  }
}
