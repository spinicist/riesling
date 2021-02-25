#include "log.h"

#include "io_nifti.h"

Log::Log(Level const l)
    : out_level_{l}
{
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

void Log::vfail(fmt::string_view fstr, fmt::format_args args) const
{
  fmt::vprint(stderr, fmt::fg(fmt::terminal_color::bright_red), fstr, args);
  fmt::print(stderr, "\n");
  exit(EXIT_FAILURE);
}

void Log::progress(long const ii, long const n) const
{
  if ((out_level_ >= Level::Info)) {
    constexpr long steps = 10;
    long const step = n / steps;
    if (ii % step == 0) {
      float progress = (100.f * ii) / n;
      fmt::print(stderr, FMT_STRING("{:.0f}%\n"), progress);
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

void Log::image(R3 const &img, std::string const &name) const
{
  if ((out_level_ >= Level::Images)) {
    WriteNifti(Info(), img, name, *this);
  }
}
