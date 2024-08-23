#include "io/hd5.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include "traj_spirals.hpp"
#include "trajectory.hpp"

#include <filesystem>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

void Dummy(std::filesystem::path const &fname) { HD5::Reader reader(fname); }

TEST_CASE("IO", "[io]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const      M = 4;
  auto const       matrix = Sz3{M, M, M};
  Index const      channels = 1, traces = 64, slices = 1, volumes = 2;
  auto const       points = ArchimedeanSpiral(M, 1.f, traces);
  Trajectory const traj(points, matrix);
  Cx5              refData(channels, points.dimension(1), points.dimension(2), slices, volumes);
  refData.setConstant(1.f);

  SECTION("Basic")
  {
    std::filesystem::path const fname("test.h5");
    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      CHECK_NOTHROW(traj.write(writer));
      CHECK_NOTHROW(writer.writeTensor<Cx, 5>(HD5::Keys::Data, refData.dimensions(), refData.data(), HD5::Dims::Noncartesian));
    }
    CHECK(std::filesystem::exists(fname));

    REQUIRE_NOTHROW(HD5::Reader(fname));
    HD5::Reader reader(fname);

    Trajectory check(reader, Eigen::Array3f::Ones());
    CHECK(traj.nSamples() == points.dimension(1));
    CHECK(traj.nTraces() == points.dimension(2));

    CHECK_NOTHROW(reader.readSlab<Cx4>(HD5::Keys::Data, {{4, 0}}));
    auto const check0 = reader.readSlab<Cx4>(HD5::Keys::Data, {{4, 0}});
    CHECK(Norm(check0 - refData.chip<4>(0)) == Approx(0.f).margin(1.e-9));
    auto const check1 = reader.readSlab<Cx4>(HD5::Keys::Data, {{4, 1}});
    CHECK(Norm(check1 - refData.chip<4>(1)) == Approx(0.f).margin(1.e-9));
    std::filesystem::remove(fname);
  }

  SECTION("Real-Data")
  { // This will now pass as I added a float->complex conversion path
    std::filesystem::path const fname("test-real.h5");

    { // Use destructor to ensure it is written
      HD5::Writer writer(fname);
      Re5 const   realData = refData.real();
      traj.write(writer);
      writer.writeTensor(HD5::Keys::Data, realData.dimensions(), realData.data(), HD5::Dims::Noncartesian);
    }
    CHECK(std::filesystem::exists(fname));
    HD5::Reader reader(fname);
    CHECK_NOTHROW(reader.readSlab<Cx4>(HD5::Keys::Data, {{0, 0}}));
    std::filesystem::remove(fname);
  }
}