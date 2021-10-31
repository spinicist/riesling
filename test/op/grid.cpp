#include "../../src/op/grid.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

TEST_CASE("ops-grid", "[ops]")
{
  //   long const M = 16;
  //   Info const info{.type = Info::Type::ThreeD,
  //                   .channels = nchan.Get(),
  //                   .matrix = Eigen::Array3l::Constant(M),
  //                   .read_points = M / 2,
  //                   .read_gap = 0,
  //                   .spokes_hi = M * M,
  //                   .spokes_lo = 0,
  //                   .lo_scale = 1.f,
  //                   .volumes = 1,
  //                   .echoes = 1,
  //                   .tr = 1.f,
  //                   .voxel_size = Eigen::Array3f::Constant(1.f),
  //                   .origin = Eigen::Array3f::Constant(0.f),
  //                   .direction = Eigen::Matrix3f::Identity()};
  //   points = ArchimedeanSpiral(info);

  //   // With credit to PyLops
  //   SECTION("Dot Test")
  //   {
  //     Cx3 x(mapSz, mapSz, mapSz), u(mapSz, mapSz, mapSz);
  //     Cx4 maps(channels, mapSz, mapSz, mapSz), y(channels, gridSz, gridSz, gridSz),
  //         v(channels, gridSz, gridSz, gridSz);

  //     v.setRandom();
  //     u.setRandom();
  //     // The maps need to be normalized for the Dot test
  //     maps.setRandom();
  //     Cx3 rss = ConjugateSum(maps, maps).sqrt();
  //     maps = maps / Tile(rss, channels);

  //     SenseOp sense(maps, y.dimensions());
  //     sense.A(u, y);
  //     sense.Adj(v, x);

  //     auto const yy = Dot(y, v);
  //     auto const xx = Dot(u, x);
  //     CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  //   }
}