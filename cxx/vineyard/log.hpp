#pragma once

#define FMT_DEPRECATED_OSTREAM

#include <chrono>
#include <exception>
#include <fmt/color.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "io/hd5-core.hpp"
#include "types.hpp"

#define LOG_DEBUG(...)                                                                                                         \
  if (rl::Log::CurrentLevel() == rl::Log::Level::Debug) { rl::Log::Print<Log::Level::High>(__VA_ARGS__); }

namespace rl {
namespace Log {

enum struct Level
{
  Testing = -1, // Suppress everything, even failures
  None = 0,
  Ephemeral = 1,
  Standard = 2,
  Debug = 3
};

class Failure : public std::runtime_error
{
public:
  Failure(std::string const &msg);
};

using Time = std::chrono::high_resolution_clock::time_point;

Level CurrentLevel();
void  SetLevel(Level const l);
void  SetDebugFile(std::string const &fname);
void  SaveEntry(std::string const &s, fmt::terminal_color const color, Level const level);
auto  Saved() -> std::string const &;
void  End();
auto  TheTime() -> std::string;

template <typename... Args> inline auto Format(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args) -> std::string
{
  return fmt::format("[{}] [{}] {}\n", TheTime(), category, fmt::format(fstr, std::forward<Args>(args)...));
}

template <typename... Args> inline void Print(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(Format(category, fstr, std::forward<Args>(args)...), fmt::terminal_color::white, Level::Ephemeral);
  }
}

template <typename... Args> inline void Debug(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(Format(category, fstr, std::forward<Args>(args)...), fmt::terminal_color::white, Level::Debug);
  }
}

template <typename... Args> inline void Warn(std::string const &category, fmt::format_string<Args...> const &fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(Format(category, fstr, std::forward<Args>(args)...), fmt::terminal_color::bright_red, Level::None);
  }
}

template <typename... Args> __attribute__((noreturn)) inline void Fail(std::string const &category, fmt::format_string<Args...> const &fstr, Args &&...args)
{
  auto const msg = Format(category, fstr, std::forward<Args>(args)...);
  if (CurrentLevel() > Level::Testing) { SaveEntry(msg, fmt::terminal_color::bright_red, Level::None); }
  throw Failure(msg);
}

template <typename... Args> __attribute__((noreturn)) inline void Fail2(fmt::format_string<Args...> const &fstr, Args &&...args)
{
  auto const msg = Format("except", fstr, std::forward<Args>(args)...);
  if (CurrentLevel() > Level::Testing) { SaveEntry(msg, fmt::terminal_color::bright_red, Level::None); }
  exit(EXIT_FAILURE);
}

void StartProgress(Index const counst, std::string const &label);
void StopProgress();
void Tick();
auto Now() -> Time;
auto ToNow(Time const t) -> std::string;

template <typename Scalar, int ND>
void Tensor(std::string const             &name,
            Sz<ND> const                  &shape,
            Scalar const                  *data,
            HD5::DimensionNames<ND> const &dims = HD5::DimensionNames<ND>());

} // namespace Log
} // namespace rl
