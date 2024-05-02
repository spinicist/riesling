#pragma once

#include "info.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
#include "types.hpp"

namespace rl {

struct Trajectory
{
  Trajectory(Re3 const &points, Eigen::Array3f const voxel_size = Eigen::Array3f::Ones());
  Trajectory(Re3 const &points, Sz3 const matrix, Eigen::Array3f const voxel_size = Eigen::Array3f::Ones());
  Trajectory(HD5::Reader &file, Eigen::Array3f const voxel_size);
  void write(HD5::Writer &file) const;
  auto nDims() const -> Index;
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  void checkDims(Sz3 const dims) const;
  auto compatible(Trajectory const &other) const -> bool;
  auto matrix() const -> Sz3;
  auto voxelSize() const -> Eigen::Array3f;
  auto FOV() const -> Eigen::Array3f;
  auto matrixForFOV(float const fov = -1.f) const -> Sz3;
  auto matrixForFOV(Eigen::Array3f const fov) const -> Sz3;
  void shiftFOV(Eigen::Vector3f const, Cx5 &data);
  auto point(int16_t const sample, int32_t const trace) const -> Re1;
  auto points() const -> Re3 const &;
  auto downsample(Eigen::Array3f const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Index, Index>;
  auto
  downsample(Cx4 const &ks, Eigen::Array3f const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Cx4>;
  auto
  downsample(Cx5 const &ks, Eigen::Array3f const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
    -> std::tuple<Trajectory, Cx5>;

private:
  void init();

  Re3            points_;
  Sz3            matrix_;
  Eigen::Array3f voxel_size_ = Eigen::Vector3f::Ones();
};

} // namespace rl
