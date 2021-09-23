#include "apodizer.h"
#include "coils.h"
#include "cropper.h"
#include "fft_plan.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "phantom_shepplogan.h"
#include "phantom_sphere.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"
#include "traj_archimedean.h"
#include "types.h"
#include <filesystem>

int main_phantom(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Filename to write phantom data to");
  args::ValueFlag<std::string> suffix(
      parser, "SUFFIX", "Add suffix (well, infix) to output dirs", {"suffix"});
  args::ValueFlag<float> grid_samp(
      parser, "SRATE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::ValueFlag<float> fov(
      parser, "FOV", "Field of View in mm (default 256)", {'f', "fov"}, 240.f);
  args::ValueFlag<long> matrix(parser, "MATRIX", "Matrix size (default 128)", {'m', "matrix"}, 128);
  args::Flag shepplogan(parser, "SHEPP-LOGAN", "3D Shepp-Logan phantom", {"shepp_logan"});
  args::ValueFlag<float> phan_r(
      parser, "RADIUS", "Radius of the spherical phantom in mm (default 90)", {"phan_rad"}, 90.f);
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> phan_c(
      parser, "X,Y,Z", "Center position of phantom (in mm)", {"center"}, Eigen::Vector3f::Zero());
  args::ValueFlag<Eigen::Vector3f, Vector3fReader> phan_rot(
      parser, "ax,ay,az", "Rotation of phantom (in deg)", {"rotation"}, Eigen::Vector3f::Zero());
  args::ValueFlag<long> coil_rings(
      parser, "COIL RINGS", "Number of rings in coil (default 1)", {"rings"}, 1);
  args::ValueFlag<float> coil_r(
      parser, "COIL RADIUS", "Radius of the coil in mm (default 150)", {"coil_rad"}, 150.f);
  args::ValueFlag<float> read_samp(
      parser, "SRATE", "Read-out oversampling (default 2)", {'r', "read"}, 2);
  args::ValueFlag<float> spoke_samp(
      parser, "SRATE", "Spoke undersampling (default 1)", {'s', "spokes"}, 1);
  args::ValueFlag<long> lores(
      parser, "LO-RES", "Include lo-res k-space with scale factor (suggest 8)", {'l', "lores"}, 0);
  args::ValueFlag<long> gap(parser, "DEAD-TIME", "Dead-time gap in read samples", {"gap"}, 0);
  args::ValueFlag<long> nchan(
      parser, "CHANNELS", "Number of channels (default 12)", {'c', "channels"}, 12);
  args::ValueFlag<std::string> sense(
      parser, "PATH", "File to read sensitivity maps from", {"sense"});
  args::ValueFlag<float> intensity(
      parser, "INTENSITY", "Phantom intensity (default 1000)", {'i', "intensity"}, 1000.f);
  args::ValueFlag<float> snr(parser, "SNR", "Add noise (specified as SNR)", {'n', "snr"}, 0);
  args::ValueFlag<std::string> trajfile(
      parser, "TRAJ FILE", "Input HD5 file for trajectory", {"traj"});
  args::ValueFlag<std::string> infofile(parser, "INFO FILE", "Input HD5 file for info", {"info"});
  args::Flag kb(parser, "KAISER-BESSEL", "Use Kaiser-Bessel kernel", {'k', "kb"});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, grid_samp.Get(), true) : (Kernel *)new NearestNeighbour();

  R3 points;
  Info info;
  bool use_lores;
  if (trajfile) {
    log.info("Reading external trajectory from {}", trajfile.Get());
    HD5::Reader reader(trajfile.Get(), log);
    Trajectory const ext_traj = reader.readTrajectory();
    info = ext_traj.info();
    points = ext_traj.points().slice(
        Sz3{0, 0, info.spokes_lo}, Sz3{3, info.read_points, info.spokes_hi});
    use_lores = info.spokes_lo > 0;
    info.spokes_lo = 0;
  } else {
    auto const spokes_hi = std::lrint(matrix.Get() * matrix.Get() / spoke_samp.Get());
    info = Info{.type = Info::Type::ThreeD,
                .channels = nchan.Get(),
                .matrix = Eigen::Array3l::Constant(matrix.Get()),
                .read_points = (long)read_samp.Get() * matrix.Get() / 2,
                .read_gap = 0,
                .spokes_hi = spokes_hi,
                .spokes_lo = 0,
                .lo_scale = lores ? lores.Get() : 1.f,
                .volumes = 1,
                .echoes = 1,
                .tr = 1.f,
                .voxel_size = Eigen::Array3f::Constant(fov.Get() / matrix.Get()),
                .origin = Eigen::Array3f::Constant(-fov.Get() / 2.f),
                .direction = Eigen::Matrix3f::Identity()};
    points = ArchimedeanSpiral(info);
    use_lores = lores;
  }
  log.info(
      FMT_STRING("Matrix Size: {} Voxel Size: {}"),
      info.matrix.transpose(),
      info.voxel_size.transpose());
  log.info(FMT_STRING("Hi-res spokes: {}"), info.spokes_hi);

  Cx4 sense_maps = sense ? InterpSENSE(sense.Get(), info.matrix, log)
                         : birdcage(
                               info.matrix,
                               info.voxel_size,
                               info.channels,
                               coil_rings.Get(),
                               coil_r.Get(),
                               coil_r.Get(),
                               log);
  info.channels = sense_maps.dimension(0); // InterpSENSE may have changed this

  Trajectory traj(info, points, log);
  Gridder hi_gridder(traj.mapping(grid_samp.Get(), kernel->radius()), kernel, false, log);
  Cx4 grid = hi_gridder.newMultichannel(info.channels);
  FFT::ThreeDMulti fft(grid, log); // FFTW needs temp space for planning

  Cropper cropper(hi_gridder.gridDims(), info.matrix, log);
  Apodizer apodizer(kernel, hi_gridder.gridDims(), cropper.size(), log);

  Cx3 phan =
      shepplogan
          ? SheppLoganPhantom(
                info.matrix,
                info.voxel_size,
                phan_c.Get(),
                phan_rot.Get(),
                phan_r.Get(),
                intensity.Get(),
                log)
          : SphericalPhantom(
                info.matrix, info.voxel_size, phan_c.Get(), phan_r.Get(), intensity.Get(), log);

  apodizer.deapodize(phan); // Don't ask me why this isn't apodize, but it works

  log.info("Generating Cartesian k-space...");
  grid.setZero();
  cropper.crop4(grid).device(Threads::GlobalDevice()) = sense_maps * Tile(phan, info.channels);

  fft.forward(grid);

  log.info("Sampling hi-res non-cartesian");
  Cx3 radial = info.noncartesianVolume();
  hi_gridder.toNoncartesian(grid, radial);

  if (use_lores) {
    Info lo_info;
    R3 lo_points;
    long lowres_scale;
    if (trajfile) {
      log.info("Reading external trajectory from {}", trajfile.Get());
      HD5::Reader reader(trajfile.Get(), log);
      Trajectory const ext_traj = reader.readTrajectory();
      lo_info = ext_traj.info();
      lo_points =
          ext_traj.points().slice(Sz3{0, 0, 0}, Sz3{3, lo_info.read_points, lo_info.spokes_lo});
      lo_info.spokes_hi = lo_info.spokes_lo;
      lo_info.spokes_lo = 0;
      lowres_scale = lo_info.lo_scale;
      lo_info.lo_scale = 1.f;
    } else {
      // Gridder does funky stuff to merge k-spaces. Sample lo-res as if it was hi-res
      lowres_scale = lores.Get();
      auto const spokes_lo = info.spokes_hi / lowres_scale;
      lo_info = Info{.type = Info::Type::ThreeD,
                     .channels = info.channels,
                     .matrix = info.matrix,
                     .read_points = info.read_points,
                     .read_gap = 0,
                     .spokes_hi = spokes_lo,
                     .spokes_lo = 0,
                     .lo_scale = 1.f,
                     .volumes = 1,
                     .echoes = 1,
                     .tr = 1.f,
                     .voxel_size = info.voxel_size,
                     .origin = info.origin,
                     .direction = Eigen::Matrix3f::Identity()};
      lo_points = ArchimedeanSpiral(lo_info);
    }
    Trajectory lo_traj(
        lo_info,
        R3(lo_points / lo_points.constant(lowres_scale)), // Points need to be scaled down here
        log);
    Gridder lo_gridder(lo_traj.mapping(grid_samp.Get(), kernel->radius()), kernel, false, log);
    Cx3 lo_radial = lo_info.noncartesianVolume();
    lo_gridder.toNoncartesian(grid, lo_radial);
    // Combine
    Cx3 const all_radial = lo_radial.concatenate(radial, 2);
    radial = all_radial;
    R3 const all_points = lo_points.concatenate(points, 2);
    points = all_points;
    info.spokes_lo = lo_info.spokes_hi;
    info.lo_scale = lowres_scale;
    traj = Trajectory(info, points, log);
  }

  if (snr) {
    float const level = intensity.Get() / (info.channels * sqrt(snr.Get()));
    log.info(FMT_STRING("Adding noise effective level {}"), level);
    Cx3 noise(radial.dimensions());
    noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
    radial += noise * noise.constant(level);
  }

  if (gap) {
    info.read_gap = gap.Get();
    radial.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_gap, info.spokes_total()}).setZero();
    traj = Trajectory(info, traj.points(), log);
  }

  HD5::Writer writer(std::filesystem::path(iname.Get()).replace_extension(".h5").string(), log);
  writer.writeTrajectory(traj);
  writer.writeNoncartesian(
      radial.reshape(Sz4{info.channels, info.read_points, info.spokes_total(), 1}));
  FFT::End(log);
  return EXIT_SUCCESS;
}
