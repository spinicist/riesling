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

using Time = std::chrono::high_resolution_clock::time_point;

Level CurrentLevel();
void  SetLevel(Level const l);
void  SetDebugFile(std::string const &fname);
auto  FormatEntry(std::string const &category, fmt::string_view fmt, fmt::format_args args) -> std::string;
void  SaveEntry(std::string const &entry, fmt::terminal_color const color, Level const level);
auto  Saved() -> std::vector<std::string> const &;
void  End();

template <typename... Args> inline void Print(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::terminal_color::white, Level::Ephemeral);
  }
}

template <typename... Args> inline void Debug(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::terminal_color::white, Level::Debug);
  }
}

template <typename... Args>
inline void Warn(std::string const &category, fmt::format_string<Args...> const &fstr, Args &&...args)
{
  if (CurrentLevel() > Level::Testing) {
    SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::terminal_color::bright_red, Level::None);
  }
}

struct Failure : std::runtime_error
{
  template <typename... Args>
  Failure(std::string const &cat, fmt::format_string<Args...> fs, Args &&...args)
    : std::runtime_error(FormatEntry(cat, fs, fmt::make_format_args(args...)))
  {
  }
};

inline void Fail(Failure const &f)
{
  if (CurrentLevel() > Level::Testing) { SaveEntry(f.what(), fmt::terminal_color::bright_red, Level::None); }
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
