#pragma once

#include <chrono>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include "types.h"

struct Log
{
  enum struct Level
  {
    None = 0,
    Info = 1,
    Images = 2,
    Debug = 3
  };

  using Time = std::chrono::high_resolution_clock::time_point;

  Log(Level const l = Level::None);

  Level level() const;

  template <typename S, typename... Args>
  inline void info(const S &fmt_str, const Args &... args) const
  {
    vinfo(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
  }

  template <typename S, typename... Args>
  inline void debug(const S &fmt_str, const Args &... args) const
  {
    vdebug(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
  }

  template <typename S, typename... Args>
  static void Fail(const S &fmt_str, const Args &... args)
  {
    vfail(fmt_str, fmt::make_args_checked<Args...>(fmt_str, args...));
  }

  void progress(long const ii, long const lo, long const hi) const;
  Time now() const;
  std::string toNow(Time const t) const;

  void image(Cx3 const &img, std::string const &name) const;
  void image(Cx4 const &img, std::string const &name) const;
  void image(R3 const &img, std::string const &name) const;

private:
  void vinfo(fmt::string_view format, fmt::format_args args) const;
  void vdebug(fmt::string_view format, fmt::format_args args) const;
  static void vfail(fmt::string_view format, fmt::format_args args);
  Level out_level_;
};

template <typename T>
inline decltype(auto) Dims(T const &x)
{
  return fmt::join(x.dimensions(), ",");
}