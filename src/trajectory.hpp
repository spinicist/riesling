#pragma once

#include "info.hpp"
#include "log.hpp"
#include "types.h"

namespace rl {

struct Trajectory
{
  Trajectory();
  Trajectory(Info const &info, Re3 const &points);
  Trajectory(Info const &info, Re3 const &points, I1 const &frames);
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  auto nFrames() const -> Index;
  auto info() const -> Info const &;
  auto point(int16_t const sample, int32_t const trace) const -> Re1;
  auto points() const -> Re3 const &;
  auto frame(Index const trace) const -> Index;
  auto frames() const -> I1 const &;
  auto downsample(float const res, Index const lores, bool const shrink) const -> std::tuple<Trajectory, Index, Index>;
  auto downsample(Cx3 const &ks, float const res, Index const lores, bool const shrink) const -> std::tuple<Trajectory, Cx3>;
  auto downsample(Cx4 const &ks, float const res, Index const lores, bool const shrink) const -> std::tuple<Trajectory, Cx4>;

private:
  void init();

  Info info_;
  Re3 points_;
  I1 frames_;
};

} // namespace rl
