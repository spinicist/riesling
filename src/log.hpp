#pragma once

#include <chrono>
#include <exception>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include "types.h"

#define LOG_DEBUG(...)                                                                                                 \
  if (rl::Log::CurrentLevel() == rl::Log::Level::High) {                                                               \
    rl::Log::Print<Log::Level::High>(__VA_ARGS__);                                                                     \
  }

namespace rl {
namespace Log {

enum struct Level
{
  Testing = -1, // Suppress everything, even failures
  None = 0,
  Low = 1,
  High = 2,
  Debug = 3
};

class Failure : public std::runtime_error
{
public:
  Failure(std::string const &msg);
};

using Time = std::chrono::high_resolution_clock::time_point;

Level CurrentLevel();
void SetLevel(Level const l);
void SetDebugFile(std::string const &fname);
void End();
auto TheTime() -> std::string;

template <Log::Level level = Log::Level::Low, typename S, typename... Args>
inline void Print(S const &fstr, Args &&...args)
{
  if (level <= CurrentLevel()) {
    fmt::print(stderr, FMT_STRING("{} {}\n"), TheTime(), fmt::format(fstr, std::forward<Args>(args)...));
  }
}

template <typename S, typename... Args>
__attribute__((noreturn)) inline void Fail(S const &fstr, Args &&...args)
{
  auto const msg =
    fmt::format(FMT_STRING("{} {}"), fmt::format("{}", TheTime()), fmt::format(fstr, std::forward<Args>(args)...));
  if (CurrentLevel() > Level::Testing) {
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", msg);
  }
  throw Failure(msg);
}

void StartProgress(Index const counst, std::string const &label);
void StopProgress();
void Tick();
Time Now();
std::string ToNow(Time const t);

template <typename Scalar, int ND>
void Tensor(Eigen::Tensor<Scalar, ND> const &i, std::string const &name);

} // namespace Log
} // namespace rl
