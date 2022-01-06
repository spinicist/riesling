
#include "coils.h"
#include "io.h"
#include "log.h"
#include "op/grid-basis.h"
#include "op/grid.h"
#include "op/recon.h"
#include "parse_args.h"
#include "phantom_shepplogan.h"
#include "phantom_sphere.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"
#include "traj_spirals.h"
#include "types.h"
#include <filesystem>

int main_phantom(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Filename to write phantom data to");
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Filename with basis", {'b', "basis"});
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f);
  std::unordered_map<std::string, Kernels> kernelMap{
    {"NN", Kernels::NN}, {"KB3", Kernels::KB3}, {"KB5", Kernels::KB5}};
  args::MapFlag<std::string, Kernels> kernel(
    parser, "K", "Choose kernel - NN, KB3, KB5", {'k', "kernel"}, kernelMap);
  args::ValueFlag<float> fov(
    parser, "FOV", "Field of View in mm (default 256)", {'f', "fov"}, 240.f);
  args::ValueFlag<Index> matrix(
    parser, "MATRIX", "Matrix size (default 128)", {'m', "matrix"}, 128);
  args::Flag shepplogan(parser, "SHEPP-LOGAN", "3D Shepp-Logan phantom", {"shepp_logan"});
  args::ValueFlag<float> phan_r(
    parser, "RADIUS", "Radius of the spherical phantom in mm (default 90)", {"phan_rad"}, 90.f);
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> phan_c(
    parser, "X,Y,Z", "Center position of phantom (in mm)", {"center"}, Eigen::Vector3f::Zero());
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> phan_rot(
    parser, "ax,ay,az", "Rotation of phantom (in deg)", {"rotation"}, Eigen::Vector3f::Zero());
  args::ValueFlag<Index> coil_rings(
    parser, "COIL RINGS", "Number of rings in coil (default 1)", {"rings"}, 1);
  args::ValueFlag<float> coil_r(
    parser, "COIL RADIUS", "Radius of the coil in mm (default 150)", {"coil_rad"}, 150.f);
  args::ValueFlag<float> read_samp(parser, "S", "Read-out oversampling (2)", {'r', "read"}, 2);
  args::ValueFlag<Index> sps(parser, "S", "Spokes per segment", {"sps"}, 256);
  args::ValueFlag<float> nex(parser, "N", "NEX (Spoke sampling rate)", {'n', "nex"}, 1);
  args::ValueFlag<Index> lores(parser, "L", "Add lo-res k-space scaled by L", {'l', "lores"}, 0);
  args::ValueFlag<Index> gap(parser, "DEAD-TIME", "Dead-time gap in read samples", {"gap"}, 0);
  args::ValueFlag<Index> nchan(parser, "C", "Number of channels (8)", {'c', "channels"}, 8);
  args::ValueFlag<std::string> sense(parser, "S", "Read SENSE maps from file", {"sense"});
  args::ValueFlag<std::vector<float>, VectorReader> intFlag(
    parser, "I", "Phantom intensities (default all 100)", {'i', "intensities"});
  args::ValueFlag<float> snr(parser, "SNR", "Add noise (specified as SNR)", {'n', "snr"}, 0);
  args::Flag phyllo(parser, "P", "Use a phyllotaxis", {'p', "phyllo"});
  args::ValueFlag<Index> smoothness(parser, "S", "Phyllotaxis smoothness", {"smoothness"}, 10);
  args::ValueFlag<Index> spi(parser, "N", "Phyllotaxis segments per interleave", {"spi"}, 4);
  args::Flag gmeans(parser, "N", "Golden-Means phyllotaxis", {"gmeans"});
  args::ValueFlag<std::string> trajfile(
    parser, "TRAJ FILE", "Input HD5 file for trajectory", {"traj"});
  args::ValueFlag<std::string> infofile(parser, "INFO FILE", "Input HD5 file for info", {"info"});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  R3 points;
  Info info;
  if (trajfile) {
    log.info("Reading external trajectory from {}", trajfile.Get());
    HD5::Reader reader(trajfile.Get(), log);
    Trajectory const ext_traj = reader.readTrajectory();
    info = ext_traj.info();
    points = ext_traj.points().slice(Sz3{0, 0, info.spokes}, Sz3{3, info.read_points, info.spokes});
  } else {
    // Follow the GE definition where factor of PI is ignored
    auto const spokes =
      sps.Get() *
      ((std::lrint(nex.Get() * matrix.Get() * matrix.Get()) + sps.Get() - 1) / sps.Get());
    info = Info{
      .type = Info::Type::ThreeD,
      .matrix = Eigen::Array3l::Constant(matrix.Get()),
      .channels = nchan.Get(),
      .read_points = (Index)read_samp.Get() * matrix.Get() / 2,
      .spokes = spokes,
      .volumes = 1,
      .echoes = 1,
      .tr = 1.f,
      .voxel_size = Eigen::Array3f::Constant(fov.Get() / matrix.Get()),
      .origin = Eigen::Array3f::Constant(-fov.Get() / 2.f),
      .direction = Eigen::Matrix3f::Identity()};
    log.info(FMT_STRING("Using {} hi-res spokes"), info.spokes);
    if (phyllo) {
      points =
        Phyllotaxis(info.read_points, info.spokes, smoothness.Get(), sps.Get() * spi.Get(), gmeans);
    } else {
      points = ArchimedeanSpiral(info.read_points, info.spokes);
    }

    if (lores) {
      auto const loMat = matrix.Get() / lores.Get();
      auto const loSpokes =
        sps.Get() * ((std::lrint(nex.Get() * loMat * loMat) + sps.Get() - 1) / sps.Get());
      auto loPoints = ArchimedeanSpiral(info.read_points, loSpokes);
      loPoints = loPoints / loPoints.constant(lores.Get());
      points = R3(loPoints.concatenate(points, 2));
      info.spokes += loSpokes;
      log.info(FMT_STRING("Added {} lo-res spokes"), loSpokes);
    }
  }
  log.info(
    FMT_STRING("Matrix Size: {} Voxel Size: {}"),
    info.matrix.transpose(),
    info.voxel_size.transpose());
  log.info(FMT_STRING("Read points: {} Spokes: {}"), info.read_points, info.spokes);

  Trajectory traj(info, points, log);
  Cx4 senseMaps = sense ? InterpSENSE(sense.Get(), info.matrix, log)
                        : birdcage(
                            info.matrix,
                            info.voxel_size,
                            info.channels,
                            coil_rings.Get(),
                            coil_r.Get(),
                            coil_r.Get(),
                            log);
  info.channels = senseMaps.dimension(0); // InterpSENSE may have changed this

  std::unique_ptr<GridBase> gridder;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    gridder = make_grid_basis(traj, osamp.Get(), kernel.Get(), false, basis, log);
  } else {
    gridder = make_grid(traj, osamp.Get(), kernel.Get(), false, log);
  }
  ReconOp recon(gridder.get(), senseMaps, log);
  auto const sz = recon.inputDimensions();
  Cx4 phan(sz);

  std::vector<float> intensities = intFlag.Get();
  if ((Index)intensities.size() == 0) {
    intensities.resize(phan.dimension(0));
    std::fill(intensities.begin(), intensities.end(), 100.f);
  } else if ((Index)intensities.size() != phan.dimension(0)) {
    Log::Fail(
      "Number of intensities {} does not match phantom first dimension {}",
      intensities.size(),
      phan.dimension(0));
  }
  for (Index ii = 0; ii < phan.dimension(0); ii++) {
    phan.chip<0>(ii) =
      shepplogan
        ? SheppLoganPhantom(
            info.matrix,
            info.voxel_size,
            phan_c.Get(),
            phan_rot.Get(),
            phan_r.Get(),
            intensities[ii],
            log)
        : SphericalPhantom(
            info.matrix, info.voxel_size, phan_c.Get(), phan_r.Get(), intensities[ii], log);
  }
  log.info("Sampling hi-res non-cartesian");
  Cx3 radial = info.noncartesianVolume();
  recon.A(phan, radial);

  if (snr) {
    float avg = std::reduce(intensities.begin(), intensities.end()) / intensities.size();
    float const level = avg / (info.channels * sqrt(snr.Get()));
    log.info(FMT_STRING("Adding noise effective level {}"), level);
    Cx3 noise(radial.dimensions());
    noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
    radial += noise * noise.constant(level);
  }

  if (gap) {
    radial.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap.Get(), info.spokes}).setZero();
    traj = Trajectory(info, points, log);
  }
  HD5::Writer writer(std::filesystem::path(iname.Get()).replace_extension(".h5").string(), log);
  writer.writeTrajectory(traj);
  writer.writeTensor(
    Cx4(radial.reshape(Sz4{info.channels, info.read_points, info.spokes, 1})), "noncartesian");
  writer.writeTensor(phan, "phantom");
  FFT::End(log);
  return EXIT_SUCCESS;
}
