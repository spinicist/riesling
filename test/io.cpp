#include "io/hd5.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "traj_spirals.h"
#include "trajectory.hpp"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace rl;
using namespace Catch;

void Dummy(std::filesystem::path const &fname)
{
  HD5::Reader reader(fname);
}

TEST_CASE("IO", "[io]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 4;
  Info const info{.matrix = Sz3{M, M, M}};
  Index const channels = 1, samples = 32, traces = 64, volumes = 2;
  auto const points = ArchimedeanSpiral(samples, traces);
  Trajectory const traj(info, points);
  Cx4 refData(channels, samples, traces, volumes);
  refData.setConstant(1.f);

  SECTION("Basic")
  {
    std::filesystem::path const fname("test.h5");
    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      CHECK_NOTHROW(traj.write(writer));
      CHECK_NOTHROW(writer.writeTensor(refData, HD5::Keys::Noncartesian));
    }
    CHECK(std::filesystem::exists(fname));

    REQUIRE_NOTHROW(HD5::Reader(fname));
    HD5::Reader reader(fname);

    Trajectory check(reader);
    CHECK(traj.nSamples() == samples);
    CHECK(traj.nTraces() == traces);

    CHECK_NOTHROW(reader.readSlab<Cx3>(HD5::Keys::Noncartesian, 0));
    auto const check0 = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, 0);
    CHECK(Norm(check0 - refData.chip<3>(0)) == Approx(0.f).margin(1.e-9));
    auto const check1 = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, 1);
    CHECK(Norm(check1 - refData.chip<3>(1)) == Approx(0.f).margin(1.e-9));
    std::filesystem::remove(fname);
  }

  SECTION("Real-Data")
  {
    std::filesystem::path const fname("test-real.h5");

    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      traj.write(writer);
      writer.writeTensor(Re4(refData.real()), HD5::Keys::Noncartesian);
    }
    CHECK(std::filesystem::exists(fname));
    HD5::Reader reader(fname);
    CHECK_THROWS_AS(reader.readSlab<Cx3>(HD5::Keys::Noncartesian, 0), Log::Failure);
    std::filesystem::remove(fname);
  }
}