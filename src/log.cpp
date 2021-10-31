#include "log.h"

#include "io_nifti.h"
#include "tensorOps.h"

Log::Log(Level const l)
    : out_level_{l}
{
}

Log::Level Log::level() const
{
  return out_level_;
}

void Log::vinfo(fmt::string_view fstr, fmt::format_args args) const
{
  if (out_level_ >= Level::Info) {
    fmt::vprint(stderr, fstr, args);
    fmt::print(stderr, "\n");
  }
}

void Log::vdebug(fmt::string_view fstr, fmt::format_args args) const
{
  if (out_level_ >= Level::Debug) {
    fmt::vprint(stderr, fstr, args);
    fmt::print(stderr, "\n");
  }
}

void Log::vfail(fmt::string_view fstr, fmt::format_args args)
{
  fmt::vprint(stderr, fmt::fg(fmt::terminal_color::bright_red), fstr, args);
  fmt::print(stderr, "\n");
  exit(EXIT_FAILURE);
}

void Log::progress(long const ii, long const lo, long const hi) const
{
  if ((out_level_ >= Level::Info) && lo == 0) {
    long const N = hi - lo;
    long const steps = std::min(N, 10L);
    long const N_per_step = N / steps;
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

Log::Time Log::now() const
{
  return std::chrono::high_resolution_clock::now();
}

std::string Log::toNow(Log::Time const t1) const
{
  using ms = std::chrono::milliseconds;
  auto const t2 = std::chrono::high_resolution_clock::now();
  auto const diff = std::chrono::duration_cast<ms>(t2 - t1).count();
  return fmt::format(FMT_STRING("{} ms"), diff);
}

void Log::image(Cx3 const &img, std::string const &name) const
{
  if ((out_level_ >= Level::Images)) {
    WriteNifti(Info(), img, name, *this);
  }
}

void Log::image(Cx4 const &img, std::string const &name) const
{
  if ((out_level_ >= Level::Images)) {
    WriteNifti(Info(), Cx4(FirstToLast4(img)), name, *this);
  }
}

void Log::image(R3 const &img, std::string const &name) const
{
  if ((out_level_ >= Level::Images)) {
    WriteNifti(Info(), img, name, *this);
  }
}
