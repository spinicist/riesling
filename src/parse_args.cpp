#include "io_nifti.h"
#include "parse_args.h"
#include "threads.h"
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <scn/scn.h>

namespace {
std::unordered_map<int, Log::Level> levelMap{
    {0, Log::Level::Fail}, {1, Log::Level::Info}, {2, Log::Level::Images}, {3, Log::Level::Debug}};
} // namespace

void Vector3fReader::operator()(
    std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  float x, y, z;
  auto result = scn::scan(scn::make_view(value), "{},{},{}", x, y, z);
  if (!result) {
    fmt::print(
        stderr,
        fmt::fg(fmt::terminal_color::bright_red),
        "Could not read vector for {} from value {} because {}\n",
        name,
        value,
        result.error());
    exit(EXIT_FAILURE);
  }
  v.x() = x;
  v.y() = y;
  v.z() = z;
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
args::ValueFlag<long> nthreads(global_group, "THREADS", "Limit number of threads", {"nthreads"});

Log ParseCommand(args::Subparser &parser, args::Positional<std::string> &fname)
{
  parser.Parse();
  Log::Level const level =
      verbosity ? verbosity.Get() : (verbose ? Log::Level::Info : Log::Level::Fail);

  Log log(level);
  if (!fname) {
    log.fail("No input specified");
  }
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads);
  }
  log.info(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());
  return log;
}

std::string OutName(
    args::Positional<std::string> &inName,
    args::ValueFlag<std::string> &name,
    std::string const &suffix,
    std::string const &extension)
{
  return fmt::format(
      FMT_STRING("{}-{}.{}"),
      name ? name.Get() : std::filesystem::path(inName.Get()).replace_extension().string(),
      suffix,
      extension);
}

long SenseVolume(args::ValueFlag<long> &sFlag, long const vols)
{
  if (sFlag) {
    return std::clamp(sFlag.Get(), 0L, vols - 1);
  } else {
    return vols - 1;
  }
}

std::vector<long> WhichVolumes(long const which, long const max_volume)
{
  if (which < 0) {
    std::vector<long> volumes(max_volume);
    std::iota(volumes.begin(), volumes.end(), 0);
    return volumes;
  } else {
    std::vector<long> volume = {which};
    return volume;
  }
}

template <typename T>
void WriteVolumes(
    Info const &info, T const &vols, long const which, std::string const &fname, Log &log)
{
  long const v_st = (which > -1) ? which : 0;
  long const v_sz = (which > -1) ? 1 : vols.dimension(3);
  Sz4 st{0, 0, 0, v_st};
  Sz4 sz{vols.dimension(0), vols.dimension(1), vols.dimension(2), v_sz};
  T const out = vols.slice(st, sz);
  WriteNifti(info, out, fname, log);
}

template void WriteVolumes(
    Info const &info, R4 const &vols, long const which, std::string const &fname, Log &log);
template void WriteVolumes(
    Info const &info, Cx4 const &vols, long const which, std::string const &fname, Log &log);