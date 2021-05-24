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
      parser, "SRATE", "Spoke undersampling (default 1)", {'s', "spokes"}, 1);
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

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, grid_samp.Get(), true) : (Kernel *)new NearestNeighbour();

  auto const mtx = Array3l::Constant(matrix.Get());
  auto const vox_sz = Eigen::Array3f::Constant(fov.Get() / matrix.Get());
  auto const origin = -(mtx.cast<float>() * vox_sz) / 2.f;

  log.info(FMT_STRING("Matrix Size: {} Voxel Size: {}"), mtx.transpose(), vox_sz.transpose());

  Cx3 phan = shepplogan
                 ? SheppLoganPhantom(mtx, vox_sz, phan_c.Get(), phan_r.Get(), intensity.Get(), log)
                 : SphericalPhantom(mtx, vox_sz, phan_c.Get(), phan_r.Get(), intensity.Get(), log);
  Cx4 sense = birdcage(mtx, vox_sz, nchan.Get(), coil_rings.Get(), coil_r.Get(), coil_r.Get(), log);

  auto const spokes_hi = std::lrint(matrix.Get() * matrix.Get() / spoke_samp.Get());
  Info info{.matrix = mtx,
            .read_points = (long)read_samp.Get() * matrix.Get() / 2,
            .read_gap = 0,
            .spokes_hi = spokes_hi,
            .spokes_lo = 0,
            .lo_scale = lores ? lores.Get() : 1.f,
            .channels = nchan.Get(),
            .type = Info::Type::ThreeD,
            .voxel_size = vox_sz,
            .origin = origin};
  log.info(FMT_STRING("Hi-res spokes: {}"), info.spokes_hi);

  R3 points = ArchimedeanSpiral(info);
  Trajectory traj(info, points, log);
  Gridder hi_gridder(traj, grid_samp.Get(), kernel, false, log);
  Cx4 grid = hi_gridder.newGrid();
  FFT3N fft(grid, log); // FFTW needs temp space for planning

  Cropper cropper(hi_gridder.gridDims(), mtx, log);
  Apodizer apodizer(kernel, hi_gridder.gridDims(), cropper.size(), log);
  apodizer.deapodize(phan); // Don't ask me why this isn't apodize, but it works

  log.info("Generating Cartesian k-space...");
  grid.setZero();
  cropper.crop4(grid).device(Threads::GlobalDevice()) = sense * Tile(phan, info.channels);
  fft.forward();

  log.info("Sampling hi-res non-cartesian");
  Cx3 radial = info.noncartesianVolume();
  hi_gridder.toNoncartesian(grid, radial);

  if (lores) {
    // Gridder does funky stuff to merge k-spaces. Sample lo-res as if it was hi-res
    auto const spokes_lo = spokes_hi / lores.Get();
    Info lo_info{.matrix = mtx,
                 .read_points = info.read_points,
                 .read_gap = 0,
                 .spokes_hi = spokes_lo,
                 .spokes_lo = 0,
                 .lo_scale = 1.f,
                 .channels = nchan.Get(),
                 .type = Info::Type::ThreeD,
                 .voxel_size = vox_sz,
                 .origin = origin};
    R3 lo_points = ArchimedeanSpiral(lo_info);
    Trajectory lo_traj(
        lo_info,
        R3(lo_points / lo_points.constant(lores.Get())), // Points need to be scaled down here
        log);
    Gridder lo_gridder(lo_traj, grid_samp.Get(), kernel, false, log);
    Cx3 lo_radial = lo_info.noncartesianVolume();
    lo_gridder.toNoncartesian(grid, lo_radial);
    // Combine
    Cx3 const all_radial = lo_radial.concatenate(radial, 2);
    radial = all_radial;
    R3 const all_points = lo_points.concatenate(points, 2);
    points = all_points;
    info.spokes_lo = lo_info.spokes_hi;
    info.lo_scale = lores.Get();
    traj = Trajectory(info, points, log);
  }

  if (snr) {
    Cx3 noise(radial.dimensions());
    noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
    radial += noise * noise.constant(intensity.Get() / snr.Get());
  }

  if (gap) {
    info.read_gap = gap.Get();
    radial.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_gap, info.spokes_total()}).setZero();
    traj = Trajectory(info, traj.points(), log);
  }

  HD5::Writer writer(std::filesystem::path(fname.Get()).replace_extension(".h5").string(), log);
  writer.writeTrajectory(traj);
  writer.writeNoncartesian(
      radial.reshape(Sz4{info.channels, info.read_points, info.spokes_total(), 1}));
  FFT::End(log);
  return EXIT_SUCCESS;
}
