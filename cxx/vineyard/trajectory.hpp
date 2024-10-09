#pragma once

#include "info.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
#include "types.hpp"

namespace rl {

template <int ND> struct TrajectoryN
{
  using SzN = Sz<ND>;
  using Array = Eigen::Array<float, ND, 1>;

  TrajectoryN(Re3 const &points, Array const voxel_size = Array::Ones());
  TrajectoryN(Re3 const &points, SzN const matrix, Array const voxel_size = Array::Ones());
  TrajectoryN(HD5::Reader &file, Array const voxel_size, SzN const matrix_size = SzN());
  void write(HD5::Writer &file) const;
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  void checkDims(SzN const dims) const;
  auto compatible(TrajectoryN const &other) const -> bool;
  auto matrix(float os = 1.f) const -> SzN;
  auto matrixForFOV(Array const fov, float os = 1.f) const -> SzN;
  auto matrixForFOV(Array const fov, Index const nB, Index const nT, float os = 1.f) const -> Sz<ND + 2>;
  auto voxelSize() const -> Array;
  auto FOV() const -> Array;
  void shiftInFOV(Eigen::Vector3f const, Cx5 &data);
  auto point(int16_t const sample, int32_t const trace) const -> Eigen::Vector<float, ND>;
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
