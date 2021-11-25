#include "../../src/op/grid.h"
#include "../../src/sdc.h"
#include "../../src/tensorOps.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

TEST_CASE("ops-grid", "[ops]")
{
  Log log;
  long const M = 16;
  float const os = 2.f;
  Info const info{
    .type = Info::Type::ThreeD,
    .channels = 1,
    .matrix = Eigen::Array3l::Constant(M),
    .read_points = long(os * M / 2),
    .read_gap = 0,
    .spokes_hi = long(M * M),
    .spokes_lo = 0,
    .lo_scale = 1.f,
    .volumes = 1,
    .echoes = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info);
  Trajectory const traj(info, points, log);
  R2 const sdc = SDC::Pipe(traj, true, os, log);

  /* I don't think the classic Dot test from PyLops is applicable to gridding,
   * because it would not be correct to have a random radial k-space. The k0
   * samples should all be the same, not random. Hence instead I calculate
   * y = Adj*A*x for NN and then check if Dot(x,y) is the same as Dot(y,y).
   * Can't check for y = x because samples not on the radial spokes will be
   * missing, and checking for KB kernel is not valid because the kernel blurs
   * the grid.
   */
  SECTION("NN-Dot")
  {
    auto grid = make_grid(traj, os, Kernels::NN, false, log);
    grid->setSDC(sdc);
    Cx4 x = grid->newMultichannel(1), y = grid->newMultichannel(1);
    Cx3 r(info.channels, info.read_points, info.spokes_total());

    x.setRandom();
    grid->A(x, r);
    grid->Adj(r, y);
    auto const xy = Dot(x, y);
    auto const yy = Dot(y, y);
    CHECK(std::abs((yy - xy) / (yy + xy + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}