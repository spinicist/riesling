#include "log.h"

using ms = std::chrono::milliseconds;

Log::Log(bool const db)
    : out_level_{db ? Level::Info : Level::Fail}
{
}

void Log::vinfo(fmt::string_view fstr, fmt::format_args args) const
{
  if (out_level_ >= Level::Info) {
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

std::chrono::high_resolution_clock::time_point Log::start_time() const
{
  return std::chrono::high_resolution_clock::now();
}

void Log::stop_time(
    std::chrono::high_resolution_clock::time_point const &start, std::string const &label) const
{
  auto const stop = std::chrono::high_resolution_clock::now();
  auto const time = std::chrono::duration_cast<ms>(stop - start).count();
  info(FMT_STRING("{} {} ms"), label, time);
}