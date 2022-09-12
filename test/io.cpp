#include "io/hd5.hpp"
#include "log.h"
#include "tensorOps.hpp"
#include "traj_spirals.h"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace rl;
using namespace Catch;

void Dummy(std::filesystem::path const &fname)
{
  HD5::RieslingReader reader(fname);
}

TEST_CASE("IO", "[io]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 4;
  float const os = 2.f;
  Info const info{.matrix = Sz3{M, M, M}, .channels = 1, .samples = Index(os * M / 2), .traces = Index(M * M), .volumes = 2};
  auto const points = ArchimedeanSpiral(info.samples, info.traces);
  Trajectory const traj(info, points);
  Cx4 refData(info.channels, info.samples, info.traces, info.volumes);
  refData.setConstant(1.f);

  SECTION("Basic")
  {
    std::filesystem::path const fname("test.h5");
    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      CHECK_NOTHROW(writer.writeTrajectory(traj));
      CHECK_NOTHROW(writer.writeTensor(refData, HD5::Keys::Noncartesian));
    }
    CHECK(std::filesystem::exists(fname));

    REQUIRE_NOTHROW(HD5::RieslingReader(fname));
    HD5::RieslingReader reader(fname);

    auto const checkInfo = reader.trajectory().info();
    CHECK(checkInfo.channels == info.channels);
    CHECK(checkInfo.samples == info.samples);
    CHECK(checkInfo.traces == info.traces);
    CHECK(checkInfo.volumes == info.volumes);

    CHECK_NOTHROW(reader.noncartesian(0));
    auto const check0 = reader.noncartesian(0);
    CHECK(Norm(check0 - refData.chip<3>(0)) == Approx(0.f).margin(1.e-9));
    auto const check1 = reader.noncartesian(1);
    CHECK(Norm(check1 - refData.chip<3>(1)) == Approx(0.f).margin(1.e-9));
    std::filesystem::remove(fname);
  }

  SECTION("Bad-Dimensions")
  {
    std::filesystem::path const fname("test-dims.h5");

    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      writer.writeTrajectory(traj);
      writer.writeTensor(
        Cx4(info.channels + 1, info.samples, info.traces, info.volumes), HD5::Keys::Noncartesian);
    }
    CHECK(std::filesystem::exists(fname));
    CHECK_THROWS_AS(Dummy(fname), Log::Failure);
    std::filesystem::remove(fname);
  }

  SECTION("Real-Data")
  {
    std::filesystem::path const fname("test-real.h5");

    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      writer.writeTrajectory(traj);
      writer.writeTensor(Re4(refData.real()), HD5::Keys::Noncartesian);
    }
    CHECK(std::filesystem::exists(fname));
    HD5::RieslingReader reader(fname);
    CHECK_THROWS_AS(reader.noncartesian(0), Log::Failure);
    std::filesystem::remove(fname);
  }
}