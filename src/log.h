#pragma once

#include <chrono>
#include <exception>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include "types.h"

namespace rl {
namespace Log {

enum struct Level
{
  Testing = -1, // Suppress everything, even failures
  None = 0,
  Info = 1,
  Progress = 2,
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

void StartProgress(Index const counst, std::string const &label);
void StopProgress();
void Tick();
Time Now();
std::string ToNow(Time const t);

template <typename Scalar, int ND>
void Tensor(Eigen::Tensor<Scalar, ND> const &i, std::string const &name);

} // namespace Log
} // namespace rl
