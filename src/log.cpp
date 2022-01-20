#include "log.h"

#include "io_nifti.h"
#include "tensorOps.h"

namespace Log {

namespace {
Level log_level = Level::None;
}

Level CurrentLevel()
{
  return log_level;
}

void SetLevel(Level const l)
{
  log_level = l;
}

void lprint(fmt::string_view fstr, fmt::format_args args)
{
  if (log_level >= Level::Info) {
    fmt::vprint(stderr, fstr, args);
    fmt::print(stderr, "\n");
  }
}

void ldebug(fmt::string_view fstr, fmt::format_args args)
{
  if (log_level >= Level::Debug) {
    fmt::vprint(stderr, fstr, args);
    fmt::print(stderr, "\n");
  }
}

void lfail(fmt::string_view fstr, fmt::format_args args)
{
  fmt::vprint(stderr, fmt::fg(fmt::terminal_color::bright_red), fstr, args);
  fmt::print(stderr, "\n");
  exit(EXIT_FAILURE);
}

void Progress(Index const ii, Index const lo, Index const hi)
{
  if ((log_level >= Level::Info) && lo == 0) {
    Index const N = hi - lo;
    Index const steps = std::min(N, 10L);
    Index const N_per_step = N / steps;
    if (ii % N_per_step == 0) { // Check for div by zero
      float progress = std::min((100.f * ii) / N, 100.f);
      if (progress < ((N - 1) * N_per_step * 100.f)) {
        fmt::print(stderr, FMT_STRING("{:.0f}%..."), progress);
      } else {
        fmt::print(stderr, FMT_STRING("{:.0f}%\n"), progress);
      }
    }
  }
}

Time Now()
{
  return std::chrono::high_resolution_clock::now();
}

std::string ToNow(Log::Time const t1)
{
  using ms = std::chrono::milliseconds;
  auto const t2 = std::chrono::high_resolution_clock::now();
  auto const diff = std::chrono::duration_cast<ms>(t2 - t1).count();
  return fmt::format(FMT_STRING("{} ms"), diff);
}

void Image(Cx3 const &img, std::string const &name)
{
  if ((log_level >= Level::Images)) {
    WriteNifti(Info(), img, name);
  }
}

void Image(Cx4 const &img, std::string const &name)
{
  if ((log_level >= Level::Images)) {
    WriteNifti(Info(), Cx4(FirstToLast4(img)), name);
  }
}

void Image(R3 const &img, std::string const &name)
{
  if ((log_level >= Level::Images)) {
    WriteNifti(Info(), img, name);
  }
}

} // namespace Log
