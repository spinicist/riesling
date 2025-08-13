#include "log.hpp"

#include "debug.hpp"
#include "fmt/chrono.h"

#include <mutex>
#include <stdio.h>
#include <unistd.h>

namespace rl {
namespace Log {

namespace {
Display                  displayLevel = Display::None;
std::mutex               logMutex;
std::vector<std::string> savedEntries;

auto TheTime() -> std::string
{
  auto const t = std::time(nullptr);
  return fmt::format("{:%H:%M:%S}", fmt::localtime(t));
}
} // namespace

void SetDisplayLevel(Display const l)
{
  displayLevel = l;
  // Move the cursor one more line down so we don't erase command names etc.
  if (displayLevel == Display::Ephemeral) { fmt::print(stderr, "\n"); }
}

auto IsHigh() -> bool { return displayLevel == Display::High; }

auto FormatEntry(std::string const &category, fmt::string_view fmt, fmt::format_args args) -> std::string
{
  return fmt::format("[{}] [{:<6}] {}", TheTime(), category, fmt::vformat(fmt, args));
}

void SaveEntry(std::string const &s, fmt::text_style const style, Display const level)
{
  {
    std::scoped_lock lock(logMutex);
    savedEntries.push_back(s); // This is not thread-safe
  }
  if (displayLevel >= level) {
    if (displayLevel == Display::Ephemeral) { fmt::print(stderr, "\033[A\33[2K\r"); }
    fmt::print(stderr, style, "{}\n", s);
  }
}

auto Saved() -> std::vector<std::string> const & { return savedEntries; }

void End()
{
  EndDebugging();
  displayLevel = Display::None;
}

Time Now() { return std::chrono::high_resolution_clock::now(); }

std::string ToNow(Log::Time const t1)
{
  using ms = std::chrono::milliseconds;
  auto const t2 = std::chrono::high_resolution_clock::now();
  auto const diff = std::chrono::duration_cast<ms>(t2 - t1).count();
  auto const hours = diff / (60 * 60 * 1000);
  auto const mins = diff % (60 * 60 * 1000) / (60 * 1000);
  auto const secs = diff % (60 * 1000) / 1000;
  auto const millis = diff % 1000;
  if (hours > 0) {
    return fmt::format("{} hour{} {} minute{} {} second{}", hours, hours > 1 ? "s" : "", mins, mins > 1 ? "s" : "", secs,
                       secs > 1 ? "s" : "");
  } else if (mins > 0) {
    return fmt::format("{} minute{} {} second{}", mins, mins > 1 ? "s" : "", secs, secs > 1 ? "s" : "");
  } else if (secs > 0) {
    return fmt::format("{}.{} seconds", secs, millis);
  } else {
    return fmt::format("{} millisecond{}", diff, diff > 1 ? "s" : "");
  }
}

} // namespace Log
} // namespace rl
