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

  struct Coord
  {
    template <typename T> using Array = Eigen::Array<T, ND, 1>;
    Array<int16_t> cart;
    int16_t        sample;
    int32_t        trace;
    Array<float>   offset;
  };

  struct CoordList
  {
    template <typename T> using Array = Eigen::Array<T, ND, 1>;
    Array<int16_t>     corner;
    std::vector<Coord> coords;
  };

  TrajectoryN(Re3 const &points, Array const voxel_size = Array::Ones());
  TrajectoryN(Re3 const &points, SzN const matrix, Array const voxel_size = Array::Ones());
  TrajectoryN(HD5::Reader &file, Array const voxel_size, SzN const matrix_size = SzN());
  void write(HD5::Writer &file) const;
  auto nSamples() const -> Index;
  auto nTraces() const -> Index;
  void checkDims(SzN const dims) const;
  auto compatible(TrajectoryN const &other) const -> bool;
  auto matrix() const -> SzN;
  auto matrixForFOV(Array const fov) const -> SzN;
  auto matrixForFOV(Array const fov, Index const nB, Index const nT) const -> Sz<ND + 2>;
  auto voxelSize() const -> Array;
  auto FOV() const -> Array;
  void shiftInFOV(Eigen::Vector3f const s, Cx5 &data) const;
  void shiftInFOV(Eigen::Vector3f const s, Index const it, Index const tst, Index const tsz, Cx5 &data) const;
  void moveInFOV(Eigen::Matrix<float, ND, ND> const R, Eigen::Vector3f const s, Cx5 &data); // Will modify trajectory
  void moveInFOV(Eigen::Matrix<float, ND, ND> const R,
                 Eigen::Vector3f const              s,
                 Index const                        it,
                 Index const                        tst,
                 Index const                        tsz,
                 Cx5                               &data); // Will modify trajectory
  auto point(int16_t const sample, int32_t const trace) const -> Eigen::Vector<float, ND>;
  auto points() const -> Re3 const &;
  auto downsample(Array const tgtSize, bool const trim, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Index, Index>;
  auto downsample(Cx4 const &ks, Array const tgtSize, bool const trim, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx4>;
  auto downsample(Cx4 const &ks, Sz3 const tgtMat, bool const trim, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx4>;
  auto downsample(Cx5 const &ks, Array const tgtSize, bool const trim, bool const shrink, bool const corners) const
    -> std::tuple<TrajectoryN, Cx5>;

  auto toCoordLists(Sz<ND> const &omat, Index const kW, Index const subgridSize, bool const conj) const
    -> std::vector<CoordList>;

private:
  void init();

  Re3   points_;
  SzN   matrix_;
  Array voxel_size_ = Array::Ones();
};

using Trajectory = TrajectoryN<3>;

} // namespace rl
