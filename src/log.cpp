#include "log.hpp"

#include "fmt/chrono.h"
#include "io/hd5.hpp"
#include "tensorOps.hpp"

#include <mutex>
#include <stdio.h>
#include <unistd.h>

namespace rl {
namespace Log {

Failure::Failure(std::string const &msg)
  : std::runtime_error(msg)
{
}

namespace {
Level log_level = Level::None;
std::shared_ptr<HD5::Writer> debug_file = nullptr;
bool isTTY = false;
Index progressTarget = -1, progressCurrent = 0, progressNext = 0;
std::mutex progressMutex;
std::string progressMessage;
std::string savedLog;
} // namespace

Level CurrentLevel()
{
  return log_level;
}

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
}

void SetDebugFile(std::string const &fname)
{
  debug_file = std::make_shared<HD5::Writer>(fname);
}

void SaveEntry(std::string const &s) {
  savedLog.append(s);
}

auto Saved() -> std::string const &{
  return savedLog;
}

void End()
{
  debug_file.reset();
  log_level = Level::None;
}

auto TheTime() -> std::string
{
  auto const t = std::chrono::system_clock::now();
  return fmt::format(FMT_STRING("[{:%H:%M:%S}]"), fmt::localtime(t));
}

void StartProgress(Index const amount, std::string const &text)
{
  if (text.size() && CurrentLevel() >= Level::High) {
    progressMessage = text;
    fmt::print(stderr, FMT_STRING("{} Starting {}\n"), TheTime(), progressMessage);
  }
  if (isTTY && CurrentLevel() >= Level::Low) {
    progressTarget = amount;
    progressCurrent = 0;
    progressNext = std::floor(progressTarget / 100.f);
  }
}

void StopProgress()
{
  if (isTTY && CurrentLevel() >= Level::Low) {
    progressTarget = -1;
    fmt::print(stderr, "\r");
  }
  if (progressMessage.size() && CurrentLevel() >= Level::High) {
    fmt::print(stderr, FMT_STRING("{} Finished {}\n"), TheTime(), progressMessage);
  }
}

void Tick()
{
  if (isTTY && (progressTarget > 0)) {
    std::scoped_lock lock(progressMutex);
    progressCurrent++;
    if (progressCurrent > progressNext) {
      float const percent = (100.f * progressCurrent) / progressTarget;
      fmt::print(stderr, FMT_STRING("\x1b[2K\r{:02.0f}%"), percent);
      progressNext += std::floor(progressTarget / 100.f);
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
  auto const hours = diff / (60 * 60 * 1000);
  auto const mins = diff % (60 * 60 * 1000) / (60 * 1000);
  auto const secs = diff % (60 * 1000) / 1000;
  if (hours > 0) {
    return fmt::format(FMT_STRING("{} hours {} minutes {} seconds"), hours, mins, secs);
  } else if (mins > 0) {
    return fmt::format(FMT_STRING("{} minutes {} seconds"), mins, secs);
  } else {
    return fmt::format(FMT_STRING("{} seconds"), secs);
  }
}

template <typename Scalar, int ND>
void Tensor(std::string const &nameIn, Sz<ND> const &shape, Scalar const *data)
{
  if (debug_file) {
    Index count = 0;
    std::string name = nameIn;
    while (debug_file->exists(name)) {
      count++;
      name = fmt::format("{}-{}", nameIn, count);
    }
    debug_file->writeTensor(name, shape, data);
  }
}

template void Tensor(std::string const &, Sz<2> const &shape, float const *data);
template void Tensor(std::string const &, Sz<3> const &shape, float const *data);
template void Tensor(std::string const &, Sz<4> const &shape, float const *data);
template void Tensor(std::string const &, Sz<5> const &shape, float const *data);
template void Tensor(std::string const &, Sz<3> const &shape, Cx const *data);
template void Tensor(std::string const &, Sz<4> const &shape, Cx const *data);
template void Tensor(std::string const &, Sz<5> const &shape, Cx const *data);
template void Tensor(std::string const &, Sz<6> const &shape, Cx const *data);

} // namespace Log
} // namespace rl
