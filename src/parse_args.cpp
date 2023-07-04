#include "parse_args.hpp"
#include "io/hd5.hpp"
#include "io/writer.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <scn/scn.h>

using namespace rl;

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None}, {1, Log::Level::Low}, {2, Log::Level::High}, {3, Log::Level::Debug}};
}

void Vector3fReader::operator()(std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  float x, y, z;
  auto  result = scn::scan(value, "{},{},{}", x, y, z);
  if (!result) { Log::Fail("Could not read vector for {} from value {} because {}", name, value, result.error()); }
  v.x() = x;
  v.y() = y;
  v.z() = z;
}

template <typename T>
void VectorReader<T>::operator()(std::string const &name, std::string const &input, std::vector<T> &values)
{
  T    val;
  auto result = scn::scan(input, "{}", val);
  if (result) {
    // Values will have been default initialized. Reset
    values.clear();
    values.push_back(val);
    while ((result = scn::scan(result.range(), ",{}", val))) {
      values.push_back(val);
    }
  } else {
    Log::Fail("Could not read argument for {}", name);
  }
}

template struct VectorReader<float>;
template struct VectorReader<Index>;

void Sz2Reader::operator()(std::string const &name, std::string const &value, Sz2 &v)
{
  Index i, j;
  auto  result = scn::scan(value, "{},{}", i, j);
  if (!result) { Log::Fail("Could not read {} from '{}': {}", name, value, result.error()); }
  v = Sz2{i, j};
}

void Sz3Reader::operator()(std::string const &name, std::string const &value, Sz3 &v)
{
  Index i, j, k;
  auto  result = scn::scan(value, "{},{},{}", i, j, k);
  if (!result) { Log::Fail("Could not read {} from '{}': {}", name, value, result.error()); }
  v = Sz3{i, j, k};
}

CoreOpts::CoreOpts(args::Subparser &parser)
  : iname(parser, "F", "Input HD5 file")
  , oname(parser, "O", "Override output name", {'o', "out"})
  , basisFile(parser, "B", "Read basis from file", {"basis", 'b'})
  , ktype(parser, "K", "Choose kernel - NN/KBn/ESn (ES3)", {'k', "kernel"}, "ES3")
  , scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu")
  , osamp(parser, "O", "Grid oversampling factor (2)", {'s', "osamp"}, 2.f)
  , fov(parser, "FOV", "Final FoV in mm (default header value)", {"fov"}, -1)
  , bucketSize(parser, "B", "Gridding bucket size (32)", {"bucket-size"}, 32)
  , residImage(parser, "R", "Write residuals in image space", {"resid-image"})
  , residKSpace(parser, "R", "Write residuals in k-space", {"resid-kspace"})
  , keepTrajectory(parser, "", "Keep the trajectory in the output file", {"keep"})
{
}

args::Group                    global_group("GLOBAL OPTIONS");
args::HelpFlag                 help(global_group, "H", "Show this help message", {'h', "help"});
args::Flag                     verbose(global_group, "V", "Print logging messages to stdout", {'v', "verbose"});
args::MapFlag<int, Log::Level> verbosity(global_group, "V", "Talk more (values 0-3)", {"verbosity"}, levelMap);
args::ValueFlag<std::string>   debug(global_group, "F", "Write debug images to file", {"debug"});
args::ValueFlag<Index>         nthreads(global_group, "N", "Limit number of threads", {"nthreads"});

void SetLogging(std::string const &name)
{
  if (verbosity) {
    Log::SetLevel(verbosity.Get());
  } else if (verbose) {
    Log::SetLevel(Log::Level::Low);
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

void ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
  SetThreadCount();
  if (!iname) { throw args::Error("No input file specified"); }
}

void ParseCommand(args::Subparser &parser)
{
  parser.Parse();
  SetLogging(parser.GetCommand().Name());
  SetThreadCount();
}

auto ReadBasis(std::string const &basisFile) -> Re2
{
  if (basisFile.empty()) {
    Re2 basis(1, 1);
    basis.setConstant(1.f);
    return basis;
  } else {
    std::string fname;
    Index       which = -1;
    auto        result = scn::scan(basisFile, "{},{}", which, fname);
    if (!result) {
      HD5::Reader basisReader(basisFile);
      return basisReader.readTensor<Re2>(HD5::Keys::Basis);
    } else {
      HD5::Reader basisReader(fname);
      auto const  basis = basisReader.readTensor<Re2>(HD5::Keys::Basis);
      if (which >= 0 && which < basis.dimension(1)) {
        Re2 const b1 = basis.slice(Sz2{0, which}, Sz2{basis.dimension(0), 1});
        return b1;
      } else {
        Log::Fail("Requested basis vector {} but only {} in file {}", which, basis.dimension(1), fname);
      }
    }
  }
}

std::string OutName(std::string const &iName, std::string const &oName, std::string const &suffix, std::string const &extension)
{
  return fmt::format(
    "{}{}.{}",
    oName.empty() ? std::filesystem::path(iName).filename().replace_extension().string() : oName,
    suffix.empty() ? "" : fmt::format("-{}", suffix),
    extension);
}

void WriteOutput(
  CoreOpts                           &opts,
  rl::Cx5 const                      &img,
  std::string const                  &suffix,
  rl::Trajectory const               &traj,
  std::string const                  &log,
  rl::Cx5 const                      &residImage,
  rl::Cx5 const                      &residKSpace,
  std::map<std::string, float> const &meta)
{
  auto const  fname = OutName(opts.iname.Get(), opts.oname.Get(), suffix, "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::Image, img.dimensions(), img.data());
  writer.writeMeta(meta);
  writer.writeInfo(traj.info());
  if (opts.keepTrajectory) { writer.writeTensor(HD5::Keys::Trajectory, traj.points().dimensions(), traj.points().data()); }
  writer.writeString("log", log);
  if (opts.residImage) { writer.writeTensor(HD5::Keys::ResidualImage, residImage.dimensions(), residImage.data()); }
  if (opts.residKSpace) { writer.writeTensor(HD5::Keys::ResidualKSpace, residKSpace.dimensions(), residKSpace.data()); }
}
