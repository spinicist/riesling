#pragma once

#define FMT_DEPRECATED_OSTREAM

#include <chrono>
#include <exception>
#include <fmt/color.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

namespace rl {
namespace Log {

enum struct Display
{
  None = 0,
  Ephemeral = 1,
  Low = 2,
  High = 3
};

using Time = std::chrono::high_resolution_clock::time_point;

void SetDisplayLevel(Display const l);
auto IsHigh() -> bool;
auto FormatEntry(std::string const &category, fmt::string_view fmt, fmt::format_args args) -> std::string;
void SaveEntry(std::string const &entry, fmt::text_style const style, Display const level);
auto Saved() -> std::vector<std::string> const &;
void End();

template <typename... Args> inline void Print(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::text_style(), Display::Ephemeral);
}

template <typename... Args> inline void Debug(std::string const &category, fmt::format_string<Args...> fstr, Args &&...args)
{
  SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::text_style(), Display::High);
}

template <typename... Args>
inline void Warn(std::string const &category, fmt::format_string<Args...> const &fstr, Args &&...args)
{
  SaveEntry(FormatEntry(category, fstr, fmt::make_format_args(args...)), fmt::fg(fmt::terminal_color::bright_yellow), Display::None);
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
  SaveEntry(f.what(), fmt::fg(fmt::terminal_color::bright_red), Display::None);
}

auto Now() -> Time;
auto ToNow(Time const t) -> std::string;

} // namespace Log
} // namespace rl
