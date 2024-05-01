#include "parse_args.hpp"
#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <scn/scan.h>

using namespace rl;

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None}, {1, Log::Level::Ephemeral}, {2, Log::Level::Standard}, {3, Log::Level::Debug}};
}

void Array2fReader::operator()(std::string const &name, std::string const &value, Eigen::Array2f &v)
{
  if (auto result = scn::scan<float, float>(value, "{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
  } else {
    Log::Fail("Could not read vector for {} from value {}", name, value);
  }
}

void Array3fReader::operator()(std::string const &name, std::string const &value, Eigen::Array3f &v)
{
  if (auto result = scn::scan<float, float, float>(value, "{},{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
    v[2] = std::get<2>(result->values());
  } else {
    Log::Fail("Could not read vector for {} from value {}", name, value);
  }
}

void Vector3fReader::operator()(std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  if (auto result = scn::scan<float, float, float>(value, "{},{},{}")) {
    v[0] = std::get<0>(result->values());
    v[1] = std::get<1>(result->values());
    v[2] = std::get<2>(result->values());
  } else {
    Log::Fail("Could not read vector for {} from value {}", name, value);
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
    Log::Fail("Could not read argument for {}", name);
  }
}

template struct VectorReader<float>;
template struct VectorReader<Index>;

template <int N>
void SzReader<N>::operator()(std::string const &name, std::string const &value, Sz<N> &sz)
{
  Index ind = 0;
  if (auto result = scn::scan<Index>(value, "{}")) {
    sz[ind] = result->value();
    for (ind = 1; ind < N; ind++) {
      result = scn::scan<Index>(result->range(), ",{}");
      if (!result) { Log::Fail("Could not read {} from '{}'", name, value); }
      sz[ind] = result->value();
    }
  } else {
    Log::Fail("Could not read {} from '{}'", name, value);
  }
}

template struct SzReader<2>;
template struct SzReader<3>;
template struct SzReader<4>;

CoreOpts::CoreOpts(args::Subparser &parser)
  : iname(parser, "FILE", "Input HD5 file")
  , oname(parser, "FILE", "Output HD5 file")
  , basisFile(parser, "B", "Read basis from file", {"basis", 'b'})
  , residual(parser, "R", "Write out residual to file", {"residuals"})
  , scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu")
  , fov(parser, "FOV", "Final FoV in mm (x,y,z)", {"fov"}, Eigen::Array3f::Zero())
  , ndft(parser, "D", "Use NDFT instead of NUFFT", {"ndft"})
{
}

PreconOpts::PreconOpts(args::Subparser &parser)
  : type(parser, "P", "Pre-conditioner (none/kspace/filename)", {"precon"}, "kspace")
  , bias(parser, "BIAS", "Pre-conditioner Bias (1)", {"precon-bias", 'b'}, 1.f)
{
}

LsqOpts::LsqOpts(args::Subparser &parser) :
  its(parser, "N", "Max iterations (4)", {'i', "max-its"}, 4),
  atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f),
   btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f),
   ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f),
   λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f)
{}

args::Group    global_group("GLOBAL OPTIONS");
args::HelpFlag help(global_group, "H", "Show this help message", {'h', "help"});
args::MapFlag<int, Log::Level>
                             verbosity(global_group, "V", "Log level 0-3", {'v', "verbosity"}, levelMap, Log::Level::Standard);
args::ValueFlag<std::string> debug(global_group, "F", "Write debug images to file", {"debug"});
args::ValueFlag<Index>       nthreads(global_group, "N", "Limit number of threads", {"nthreads"});

void SetLogging(std::string const &name)
{
  if (verbosity) {
    Log::SetLevel(verbosity.Get());
  } else if (char *const env_p = std::getenv("RL_VERBOSITY")) {
    Log::SetLevel(levelMap.at(std::atoi(env_p)));
  }

  Log::Print("Welcome to RIESLING");
  Log::Print("Command: {}", name);

  if (debug) { Log::SetDebugFile(debug.Get()); }
}

void SetThreadCount()
{
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads.Get());
  } else if (char *const env_p = std::getenv("RL_THREADS")) {
    Threads::SetGlobalThreadCount(std::atoi(env_p));
  }
  Log::Print("Using {} threads", Threads::GlobalThreadCount());
}

void ParseCommand(args::Subparser &parser)
{
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
  SetThreadCount();
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

void WriteOutput(std::string const &fname, rl::Cx5 const &img, rl::Info const &info, std::string const &log)
{
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::Data, img.dimensions(), img.data(), HD5::Dims::Image);
  writer.writeInfo(info);
  if (log.size()) { writer.writeString("log", log); }
  Log::Print("Wrote output file {}", fname);
}
