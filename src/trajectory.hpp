#pragma once

#include "info.hpp"
#include "types.hpp"

namespace rl {

struct Trajectory
{
  Trajectory();
  Trajectory(Info const &info, Re3 const &points);

  auto nDims() const -> Index;
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  void checkDims(Sz3 const dims) const;
  auto compatible(Trajectory const &other) const -> bool;
  auto info() const -> Info const &;
  auto matrix() const -> Sz3;
  auto FOV() const -> Eigen::Array3f;
  auto matrixForFOV(float const fov = -1.f) const -> Sz3;
  auto matrixForFOV(Eigen::Array3f const fov) const -> Sz3; 
  auto point(int16_t const sample, int32_t const trace) const -> Re1;
  auto points() const -> Re3 const &;
  auto downsample(float const res, Index const lores, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Index, Index>;
  auto downsample(Cx4 const &ks, float const res, Index const lores, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Cx4>;
  auto downsample(Cx5 const &ks, float const res, Index const lores, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Cx5>;

private:
  void init();

  Info info_;
  Re3  points_;
};

} // namespace rl
