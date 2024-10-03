#include "log.hpp"

#include "fmt/chrono.h"
#include "io/hd5.hpp"
#include "tensors.hpp"

#include <mutex>
#include <stdio.h>
#include <unistd.h>

namespace rl {
namespace Log {

namespace {
Level                        log_level = Level::None;
std::shared_ptr<HD5::Writer> debug_file = nullptr;
bool                         isTTY = false;
std::mutex                   logMutex;
std::vector<std::string>     savedEntries;

auto TheTime() -> std::string
{
  auto const t = std::time(nullptr);
  return fmt::format("{:%H:%M:%S}", fmt::localtime(t));
}
} // namespace

Level CurrentLevel() { return log_level; }

void SetLevel(Level const l)
{
  log_level = l;
  if (char *const env_p = std::getenv("RL_NOT_TTY")) {
    isTTY = false;
  } else if (isatty(fileno(stdin))) {
    isTTY = true;
  } else {
    isTTY = false;
  }
  // Move the cursor one more line down so we don't erase command names etc.
  if (CurrentLevel() == Level::Ephemeral) { fmt::print(stderr, "\n"); }
}

void SetDebugFile(std::string const &fname) { debug_file = std::make_shared<HD5::Writer>(fname); }

auto FormatEntry(std::string const &category, fmt::string_view fmt, fmt::format_args args) -> std::string
{
  return fmt::format("[{}] [{:<6}] {}", TheTime(), category, fmt::vformat(fmt, args));
}

void SaveEntry(std::string const &s, fmt::terminal_color const color, Level const level)
{
  {
    std::scoped_lock lock(logMutex);
    savedEntries.push_back(s); // This is not thread-safe
  }
  if (CurrentLevel() >= level) {
    if (CurrentLevel() == Level::Ephemeral) { fmt::print(stderr, "\033[A\33[2K\r"); }
    fmt::print(stderr, fmt::fg(color), "{}\n", s);
  }
}

auto Saved() -> std::vector<std::string> const & { return savedEntries; }

void End()
{
  debug_file.reset();
  log_level = Level::None;
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

template <typename Scalar, int N>
void Tensor(std::string const &nameIn, Sz<N> const &shape, Scalar const *data, HD5::DimensionNames<N> const &dimNames)
{
  if (debug_file) {
    Index       count = 0;
    std::string name = nameIn;
    while (debug_file->exists(name)) {
      count++;
      name = fmt::format("{}-{}", nameIn, count);
    }
    debug_file->writeTensor(name, shape, data, dimNames);
  }
}

template void Tensor(std::string const &, Sz<1> const &shape, float const *data, HD5::DimensionNames<1> const &);
template void Tensor(std::string const &, Sz<2> const &shape, float const *data, HD5::DimensionNames<2> const &);
template void Tensor(std::string const &, Sz<3> const &shape, float const *data, HD5::DimensionNames<3> const &);
template void Tensor(std::string const &, Sz<4> const &shape, float const *data, HD5::DimensionNames<4> const &);
template void Tensor(std::string const &, Sz<5> const &shape, float const *data, HD5::DimensionNames<5> const &);
template void Tensor(std::string const &, Sz<3> const &shape, Cx const *data, HD5::DimensionNames<3> const &);
template void Tensor(std::string const &, Sz<4> const &shape, Cx const *data, HD5::DimensionNames<4> const &);
template void Tensor(std::string const &, Sz<5> const &shape, Cx const *data, HD5::DimensionNames<5> const &);
template void Tensor(std::string const &, Sz<6> const &shape, Cx const *data, HD5::DimensionNames<6> const &);

} // namespace Log
} // namespace rl
