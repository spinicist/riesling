#include "rl/basis/basis.hpp"
#include "rl/fft.hpp"
#include "rl/log/log.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/nufft-lowmem.hpp"
#include "rl/op/nufft.hpp"
#include "rl/op/sense.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Recon", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Index const M = GENERATE(7); //, 15, 16);
  Index const nC = 4;
  auto const  matrix = Sz3{M, M, M};
  Re3         points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  Trajectory const traj(points, matrix);
  Basis            basis;

  float const       osamp = GENERATE(1.3f);
  auto              nufft = TOps::MakeNUFFT<3>(GridOpts<3>{.osamp = osamp}, traj, nC, &basis);

  Cx5 senseMaps(AddBack(traj.matrix(), 1, nC));
  senseMaps.setConstant(std::sqrt(1. / nC));
  auto sense = TOps::MakeSENSE(senseMaps, 1);
  auto recon = TOps::MakeCompose(sense, nufft);

  Cx3 ks(recon->oshape);
  Cx4 img(recon->ishape);
  ks.setConstant(1.f);
  img = recon->adjoint(ks);
  // Super loose tolerance
  // INFO("ks\n" << ks);
  // INFO("img\n" << img);
  CHECK(Norm<false>(img) == Approx(Norm<false>(ks)).margin(2.e-1f));
  ks = recon->forward(img);
  // INFO("ks\n" << ks);
  CHECK(Norm<false>(ks) == Approx(Norm<false>(img)).margin(2.e-1f));
}

TEST_CASE("ReconLowmem", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Index const M = GENERATE(8); //, 15, 16);
  Index const nC = 4;
  auto const  matrix = Sz3{M, M, M};
  Re3         points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  Trajectory const traj(points, matrix);
  Basis            basis;

  Cx5 sKern(AddBack(traj.matrix(), 1, nC));
  sKern.setConstant(std::sqrt(1. / nC));
  FFT::Forward(sKern, Sz3{0, 1, 2});
  auto recon = TOps::NUFFTLowmem<3>::Make(GridOpts<3>(), traj, sKern, &basis);

  Cx3 ks(recon->oshape);
  Cx4 img(recon->ishape);
  ks.setConstant(1.f);
  img = recon->adjoint(ks);
  // Super loose tolerance
  // INFO("ks\n" << ks);
  // INFO("img\n" << img);
  CHECK(Norm<false>(img) == Approx(Norm<false>(ks)).margin(2.e-1f));
  ks = recon->forward(img);
  // INFO("ks\n" << ks);
  CHECK(Norm<false>(ks) == Approx(Norm<false>(img)).margin(2.e-1f));
}
