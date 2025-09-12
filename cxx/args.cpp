#include "args.hpp"

#include "rl/io/writer.hpp"
#include "rl/log/debug.hpp"
#include "rl/sys/threads.hpp"

#include <exception>
#include <fmt/format.h>
#include <scn/scan.h>

using namespace rl;

namespace {
std::unordered_map<int, Log::Display> levelMap{
  {0, Log::Display::None}, {1, Log::Display::Ephemeral}, {2, Log::Display::Low}, {3, Log::Display::High}};
}

args::Group                      global_group("GLOBAL OPTIONS");
args::HelpFlag                   help(global_group, "H", "Show this help message", {'h', "help"});
args::MapFlag<int, Log::Display> verbosity(global_group, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Display::Low);
args::ValueFlag<std::string>     debug(global_group, "F", "Write debug images to file", {"debug"});
args::ValueFlag<Index>           nthreads(global_group, "N", "Limit number of threads", {"nthreads"});
args::ValueFlag<Index>           deflate_level(global_group, "D", "Set deflate level (0=none)", {"deflate"}, 2);

void SetLogging(std::string const &name)
{
  if (verbosity) {
    Log::SetDisplayLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetDisplayLevel(levelMap.at(std::atoi(env_p)));
  }
  Log::Print(name, "Welcome to RIESLING");
  if (debug) { Log::SetDebugFile(debug.Get()); }
}

void SetThreadCount()
{
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads.Get());
  } else if (char *const env_p = std::getenv("RL_THREADS")) {
    Threads::SetGlobalThreadCount(std::atoi(env_p));
  }
}

void SetDeflate()
{
  if (deflate_level) {
    HD5::SetDeflate(deflate_level.Get());
  } else if (char *const env_p = std::getenv("RL_DEFLATE")) {
    HD5::SetDeflate(atoi(env_p));
  }
}

void ParseCommand(args::Subparser &parser)
{
  args::GlobalOptions globals(parser, global_group);
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
  SetThreadCount();
  SetDeflate();
}

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }
}

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname, args::Positional<std::string> &oname)
{
  ParseCommand(parser);
  if (!iname) { throw args::Error("No input file specified"); }
  if (!oname) { throw args::Error("No output file specified"); }
}

class ArgsError : public std::runtime_error
{
public:
  ArgsError(std::string const &msg)
    : std::runtime_error(msg)
  {
  }
};

template <int N> void SzReader<N>::operator()(std::string const &name, std::string const &value, rl::Sz<N> &sz)
{
  size_t ind = 0;
  if (auto result = scn::scan<Index>(value, "{}")) {
    sz[ind] = result->value();
    for (ind = 1; ind < N; ind++) {
      result = scn::scan<Index>(result->range(), ",{}");
      if (!result) { throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value))); }
      sz[ind] = result->value();
    }
  } else {
    throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value)));
  }
}

template struct SzReader<2>;
template struct SzReader<3>;
template struct SzReader<4>;
template struct SzReader<5>;
template struct SzReader<6>;

template <typename T, int ND>
void ArrayReader<T, ND>::operator()(std::string const &name, std::string const &value, Eigen::Array<T, ND, 1> &v)
{
  size_t ind = 0;
  if (auto result = scn::scan<T>(value, "{}")) {
    v[ind] = result->value();
    for (ind = 1; ind < ND; ind++) {
      result = scn::scan<T>(result->range(), ",{}");
      if (!result) { throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value))); }
      v[ind] = result->value();
    }
  } else {
    throw(ArgsError(fmt::format("Could not read {} from '{}'", name, value)));
  }
}

template struct ArrayReader<float, 1>;
template struct ArrayReader<float, 2>;
template struct ArrayReader<float, 3>;

void Vector3fReader::operator()(std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  if (auto result = scn::scan<float, float, float>(value, "{},{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
    v[2] = std::get<2>(result->values());
  } else {
    throw(ArgsError(fmt::format("Could not read vector for {} from value {}", name, value)));
  }
}

void Matrix3fReader::operator()(std::string const &name, std::string const &value, Eigen::Matrix3f &m)
{
  if (auto result =
        scn::scan<float, float, float, float, float, float, float, float, float>(value, "{},{},{},{},{},{},{},{},{}")) {
    m(0, 0) = std::get<0>(result->values());
    m(0, 1) = std::get<1>(result->values());
    m(0, 2) = std::get<2>(result->values());
    m(1, 0) = std::get<3>(result->values());
    m(1, 1) = std::get<4>(result->values());
    m(1, 2) = std::get<5>(result->values());
    m(2, 0) = std::get<6>(result->values());
    m(2, 1) = std::get<7>(result->values());
    m(2, 2) = std::get<8>(result->values());
  } else {
    throw(ArgsError(fmt::format("Could not read vector for {} from value {}", name, value)));
  }
}

void ArrayXfReader::operator()(std::string const &name, std::string const &input, Eigen::ArrayXf &val)
{
  auto               result = scn::scan<float>(input, "{}");
  std::vector<float> values;
  if (result) {
    values.push_back(result->value());
    while ((result = scn::scan<float>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
  val.resize(values.size());
  for (size_t ii = 0; ii < values.size(); ii++) {
    val[ii] = values[ii];
  }
}

void ArrayXiReader::operator()(std::string const &name, std::string const &input, Eigen::ArrayXi &val)
{
  auto             result = scn::scan<int>(input, "{}");
  std::vector<int> values;
  if (result) {
    values.push_back(result->value());
    while ((result = scn::scan<int>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
  val.resize(values.size());
  for (size_t ii = 0; ii < values.size(); ii++) {
    val[ii] = values[ii];
  }
}

template <typename T>
void VectorReader<T>::operator()(std::string const &name, std::string const &input, std::vector<T> &values)
{
  auto result = scn::scan<T>(input, "{}");
  if (result) {
    // Values will have been default initialized. Reset
    values.clear();
    values.push_back(result->value());
    while ((result = scn::scan<T>(result->range(), ",{}"))) {
      values.push_back(result->value());
    }
  } else {
    throw(ArgsError(fmt::format("Could not read argument for {}", name)));
  }
}

template struct VectorReader<float>;
template struct VectorReader<Index>;
