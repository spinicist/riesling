#include "basis/basis.hpp"
#include "fft.hpp"
#include "log.hpp"
#include "op/compose.hpp"
#include "op/nufft-lowmem.hpp"
#include "op/nufft.hpp"
#include "op/sense.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Recon-Basic", "[recon]")
{
  Log::SetLevel(Log::Level::Testing);
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
  std::string const ktype = GENERATE("ES4");
  auto              nufft = TOps::NUFFT<3>::Make(TOps::Grid<3>::Opts{.osamp = osamp, .ktype = ktype}, traj, nC, &basis);

  Cx5 senseMaps(AddFront(traj.matrix(), 1, nC));
  senseMaps.setConstant(std::sqrt(1. / nC));
  auto sense = std::make_shared<TOps::SENSE>(senseMaps, false, 1);

  TOps::Compose<TOps::SENSE, TOps::TOp<Cx, 5, 3>> recon(sense, nufft);

  Cx3 ks(recon.oshape);
  Cx4 img(recon.ishape);
  ks.setConstant(1.f);
  img = recon.adjoint(ks);
  // Super loose tolerance
  // INFO("ks\n" << ks);
  // INFO("img\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-1f));
  ks = recon.forward(img);
  // INFO("ks\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-1f));
}

TEST_CASE("Recon-Lowmem", "[recon]")
{
  Log::SetLevel(Log::Level::Testing);
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

  Cx5 sKern(AddFront(traj.matrix(), 1, nC));
  sKern.setConstant(std::sqrt(1. / nC));
  FFT::Forward(sKern, Sz3{2, 3, 4});
  auto recon = TOps::NUFFTLowmem<3>::Make(TOps::Grid<3>::Opts(), traj, sKern, &basis);

  Cx3 ks(recon->oshape);
  Cx4 img(recon->ishape);
  ks.setConstant(1.f);
  img = recon->adjoint(ks);
  // Super loose tolerance
  INFO("ks\n" << ks);
  INFO("img\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-1f));
  ks = recon->forward(img);
  INFO("ks\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-1f));
}
