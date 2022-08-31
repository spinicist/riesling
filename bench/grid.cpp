#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/info.h"
#include "../src/traj_spirals.h"
#include "../src/op/gridBase.hpp"

#include <catch2/catch.hpp>

using namespace rl;

Index const M = 128;
Index const C = 8;
Info const info{.channels = C, .samples = M / 2, .traces = Index(M * M), .matrix = Sz3{M, M, M}};
auto const points = ArchimedeanSpiral(info.samples, info.traces);
Trajectory traj(info, points);
float const os = 2.f;
// Index const bucketSz = 32;

TEST_CASE("Grid")
{
  auto gridfi3 = make_grid<Cx, 3>(traj, "ES3", os, C);
  auto gridfi5 = make_grid<Cx, 3>(traj, "ES5", os, C);

  Cx5 c(gridfi3->inputDimensions());
  Cx3 nc(gridfi3->outputDimensions());

  nc.setRandom();

  BENCHMARK("ES3 Noncartesian->Cartesian")
  {
    gridfi3->adjoint(nc);
  };

  BENCHMARK("ES5 Noncartesian->Cartesian")
  {
    gridfi5->adjoint(nc);
  };

  BENCHMARK("ES3 Cartesian->Noncartesian")
  {
    gridfi3->forward(c);
  };

  BENCHMARK("ES5 Cartesian->Noncartesian")
  {
    gridfi5->forward(c);
  };
}

TEST_CASE("GridBasisAdj")
{
  Index const nB = 4;
  Re2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridfi3 = make_grid<Cx, 3>(traj, "ES3", os, C, basis);
  auto gridfi5 = make_grid<Cx, 3>(traj, "ES5", os, C, basis);

  Cx5 c(gridfi3->inputDimensions());
  Cx3 nc(gridfi3->outputDimensions());

  BENCHMARK("ES3 Noncartesian->Cartesian")
  {
    gridfi3->adjoint(nc);
  };

  BENCHMARK("ES5 Noncartesian->Cartesian")
  {
    gridfi5->adjoint(nc);
  };

  BENCHMARK("ES3 Cartesian->Noncartesian")
  {
    gridfi3->forward(c);
  };

  BENCHMARK("ES5 Cartesian->Noncartesian")
  {
    gridfi5->forward(c);
  };
}
