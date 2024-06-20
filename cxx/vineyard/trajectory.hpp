#pragma once

#include "info.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
#include "types.hpp"

namespace rl {

template <int NDims> struct TrajectoryN
{
  using SzN = Sz<NDims>;
  using Array = Eigen::Array<float, NDims, 1>;

  TrajectoryN(Re3 const &points, Array const voxel_size = Array::Ones());
  TrajectoryN(Re3 const &points, SzN const matrix, Array const voxel_size = Array::Ones());
  TrajectoryN(HD5::Reader &file, Array const voxel_size);
  void write(HD5::Writer &file) const;
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  void checkDims(SzN const dims) const;
  auto compatible(TrajectoryN const &other) const -> bool;
  auto matrix(float os = 1.f) const -> SzN;
  auto matrixForFOV(Array const fov, float os = 1.f) const -> SzN;
  auto voxelSize() const -> Array;
  auto FOV() const -> Array;
  void shiftFOV(Eigen::Vector3f const, Cx5 &data);
  auto point(int16_t const sample, int32_t const trace) const -> Re1;
  auto points() const -> Re3 const &;
  auto downsample(Array const tgtSize,
                  Index const fullResTraces,
                  bool const  shrink,
                  bool const  corners) const -> std::tuple<TrajectoryN, Index, Index>;
  auto downsample(Cx4 const &ks, Array const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx4>;
  auto downsample(Cx4 const &ks, Sz3 const tgtMat, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx4>;
  auto downsample(Cx5 const &ks, Array const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx5>;

private:
  void init();

  Re3   points_;
  SzN   matrix_;
  Array voxel_size_ = Array::Ones();
};

using Trajectory = TrajectoryN<3>;

} // namespace rl
