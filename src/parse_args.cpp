#include "parse_args.h"
#include "io/io.h"
#include "tensorOps.h"
#include "threads.h"
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <scn/scn.h>

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None},
  {1, Log::Level::Info},
  {2, Log::Level::Progress},
  {3, Log::Level::Debug},
  {4, Log::Level::Images}};
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
args::HelpFlag help(global_group, "HELP", "Show this help message", {'h', "help"});
args::Flag verbose(global_group, "VERBOSE", "Talk more", {'v', "verbose"});
args::MapFlag<int, Log::Level> verbosity(
  global_group,
  "VERBOSITY",
  "Talk even more (values 0-3, see documentation)",
  {"verbosity"},
  levelMap);
args::ValueFlag<Index> nthreads(global_group, "THREADS", "Limit number of threads", {"nthreads"});

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  parser.Parse();
  Log::SetLevel(verbosity ? verbosity.Get() : (verbose ? Log::Level::Info : Log::Level::None));
  Log::Print(FMT_STRING("Starting: {}"), parser.GetCommand().Name());
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
