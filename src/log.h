#pragma once

#include <chrono>
#include <exception>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include "types.h"

namespace Log {
enum struct Level
{
  None = 0,
  Info = 1,
  Progress = 2,
  Debug = 3,
  Images = 4
};

class Failure : public std::runtime_error
{
public:
  Failure(std::string const &msg);
};

using Time = std::chrono::high_resolution_clock::time_point;

Level CurrentLevel();
void SetLevel(Level const l);

void lprint(fmt::string_view fstr, fmt::format_args args);
void ldebug(fmt::string_view fstr, fmt::format_args args);
__attribute__((noreturn)) void lfail(fmt::string_view fstr, fmt::format_args args);

template <typename S, typename... Args>
inline void Print(const S &fmt_str, const Args &...args)
{
  lprint(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
}

template <typename S, typename... Args>
inline void Debug(const S &fmt_str, const Args &...args)
{
  ldebug(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
}

template <typename S, typename... Args>
__attribute__((noreturn)) inline void Fail(const S &fmt_str, const Args &...args)
{
  lfail(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
}

void Progress(Index const ii, Index const lo, Index const hi);
Time Now();
std::string ToNow(Time const t);

void Image(Cx3 const &img, std::string const &name);
void Image(Cx4 const &img, std::string const &name);
void Image(R3 const &img, std::string const &name);
} // namespace Log
