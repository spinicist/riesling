#pragma once

#include <chrono>
#include <fmt/color.h>
#include <fmt/ostream.h>

struct Log
{
  enum struct Level
  {
    Fail = 0,
    Error = 1,
    Info = 2
  };

  Log(bool const db);

  template <typename S, typename... Args>
  inline void info(const S &fmt_str, const Args &... args) const
  {
    vinfo(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
  }

  template <typename S, typename... Args>
  inline void fail(const S &fmt_str, const Args &... args) const
  {
    vfail(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
  }

  std::chrono::high_resolution_clock::time_point start_time() const;
  void stop_time(std::chrono::high_resolution_clock::time_point const &t, std::string const &label) const;

private:
  void vinfo(fmt::string_view format, fmt::format_args args) const;
  void vfail(fmt::string_view format, fmt::format_args args) const;
  Level out_level_;
};