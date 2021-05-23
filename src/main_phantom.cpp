#include "apodizer.h"
#include "coils.h"
#include "cropper.h"
#include "fft3n.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "phantom_shepplogan.h"
#include "phantom_sphere.h"
#include "tensorOps.h"
#include "threads.h"
#include "traj_archimedean.h"
#include "types.h"
#include <filesystem>

int main_phantom(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "Filename to write phantom data to");
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
  args::ValueFlag<long> coil_rings(
      parser, "COIL RINGS", "Number of rings in coil (default 1)", {"rings"}, 1);
  args::ValueFlag<float> coil_r(
      parser, "COIL RADIUS", "Radius of the coil in mm (default 150)", {"coil_rad"}, 150.f);
  args::ValueFlag<float> read_samp(
      parser, "SRATE", "Read-out oversampling (default 2)", {'r', "read"}, 2);
  args::ValueFlag<float> spoke_samp(
      parser, "SRATE", "Sample factor for spokes (default 1)", {'s', "spokes"}, 1);
  args::ValueFlag<long> lores(
      parser, "LO-RES", "Include lo-res k-space with scale factor (suggest 8)", {'l', "lores"}, 0);
  args::ValueFlag<long> gap(parser, "DEAD-TIME", "Dead-time gap in read samples", {"gap"}, 0);
  args::ValueFlag<long> nchan(
      parser, "CHANNELS", "Number of channels (default 12)", {'c', "channels"}, 12);
  args::ValueFlag<float> intensity(
      parser, "INTENSITY", "Phantom intensity (default 1000)", {'i', "intensity"}, 1000.f);
  args::ValueFlag<float> snr(parser, "SNR", "Add noise (specified as SNR)", {'n', "snr"}, 0);
  args::Flag kb(parser, "KAISER-BESSEL", "Use Kaiser-Bessel kernel", {'k', "kb"});

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);

  auto const m = matrix.Get();
  auto const vox_sz = fov.Get() / m;
  auto const spokes_hi = std::lrint(spoke_samp.Get() * m * m);
  auto const o = -(m * vox_sz) / 2;
  // Strategy - sample the grid at the *grid* sampling rate, and then decimate to read sampling rate
  Info grid_info{.matrix = Array3l{m, m, m},
                 .read_points = (long)read_samp.Get() * m / 2,
                 .read_gap = 0,
                 .spokes_hi = spokes_hi,
                 .spokes_lo = lores ? static_cast<long>(spokes_hi / lores.Get()) : 0,
                 .lo_scale = lores ? lores.Get() : 1.f,
                 .channels = nchan.Get(),
                 .type = Info::Type::ThreeD,
                 .voxel_size = Eigen::Array3f{vox_sz, vox_sz, vox_sz},
                 .origin = Eigen::Vector3f{o, o, o}};
  log.info(
      FMT_STRING("Matrix Size: {} Voxel Size: {} Oversampling: {} Dead-time Gap: {}"),
      matrix.Get(),
      vox_sz,
      read_samp.Get(),
      gap.Get());
  log.info(
      FMT_STRING("Hi-res spokes: {} Lo-res spokes: {} Lo-res scale: {}"),
      grid_info.spokes_hi,
      grid_info.spokes_lo,
      grid_info.lo_scale);

  // We want effective sample positions at the middle of the bin
  Trajectory grid_traj(grid_info, ArchimedeanSpiral(grid_info, 0), log);
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, grid_samp.Get(), true) : (Kernel *)new NearestNeighbour();
  Gridder gridder(grid_traj, grid_samp.Get(), kernel, false, log);
  Cx4 grid = gridder.newGrid();
  FFT3N fft(grid, log); // FFTW needs temp space for planning

  Cropper cropper(gridder.gridDims(), grid_info.matrix, log);
  Cx3 phan;
  if (shepplogan) {
    phan = SheppLoganPhantom(grid_info, phan_c.Get(), phan_r.Get(), intensity.Get(), log);
  } else {
    phan = SphericalPhantom(grid_info, phan_c.Get(), phan_r.Get(), intensity.Get(), log);
  }
  Apodizer apodizer(kernel, gridder.gridDims(), cropper.size(), log);
  apodizer.deapodize(phan); // Don't ask me why this isn't apodize, but it works

  // Generate SENSE maps and multiply
  log.info("Generating coil sensitivities...");
  Cx4 sense = birdcage(grid_info, coil_rings.Get(), coil_r.Get(), coil_r.Get(), log);
  log.info("Generating coil images...");
  grid.setZero();
  cropper.crop4(grid).device(Threads::GlobalDevice()) = sense * Tile(phan, grid_info.channels);
  if (log.level() >= Log::Level::Images) { // Extra check to avoid the shuffle when we can
    log.image(SwapToChannelLast(grid), "phantom-prefft.nii");
  }
  log.info("FFT to k-space");
  fft.forward();
  if (log.level() >= Log::Level::Images) { // Extra check to avoid the shuffle when we can
    log.image(SwapToChannelLast(grid), "phantom-postfft.nii");
  }
  Cx3 radial = grid_info.noncartesianVolume();
  gridder.toNoncartesian(grid, radial);
  if (snr) {
    Cx3 noise(radial.dimensions());
    noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
    radial += noise * noise.constant(intensity.Get() / snr.Get());
  }

  HD5::Writer writer(std::filesystem::path(fname.Get()).replace_extension(".h5").string(), log);
  if (gap) {
    grid_info.read_gap = gap.Get();
    Sz3 sz{grid_info.channels, grid_info.read_gap, grid_info.spokes_total()};
    fmt::print("radial {} slice {}\n", fmt::join(radial.dimensions(), ","), fmt::join(sz, ","));
    radial.slice(Sz3{0, 0, 0}, sz).setZero();
    grid_traj = Trajectory(grid_info, grid_traj.points(), log);
  }
  writer.writeTrajectory(grid_traj);
  writer.writeNoncartesian(
      radial.reshape(Sz4{grid_info.channels, grid_info.read_points, grid_info.spokes_total(), 1}));

  FFT::End(log);
  return EXIT_SUCCESS;
}
