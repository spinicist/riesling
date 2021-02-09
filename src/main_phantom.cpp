#include "cropper.h"
#include "fft3n.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"
#include "traj_archimedean.h"
#include "types.h"
#include <filesystem>

/* This function is ported from ismrmd_phantom.cpp, which in turn is inspired by
 * http://web.eecs.umich.edu/~fessler/code/
 */
Cx4 birdcage(
    Dims3 const sz,
    long const nchan,
    float const coil_rad_mm,  // Radius of the actual coil, i.e. where the channels should go
    float const sense_rad_mm, // Sensitivity radius
    Info const &info)
{
  Cx4 all(nchan, sz[0], sz[1], sz[2]);
  auto const avoid_div_zero = info.voxel_size.matrix().norm() / 2.f;
  for (long ic = 0; ic < nchan; ic++) {
    auto const fc = static_cast<float>(ic) / nchan;
    float const phase = fc * 2 * M_PI;
    Eigen::Vector3f const chan_pos =
        coil_rad_mm * Eigen::Vector3f(std::cos(phase), std::sin(phase), 0.);
    for (long iz = 0; iz < sz[2]; iz++) {
      auto const pz = 0.f; // Assume strip-lines
      for (long iy = 0; iy < sz[1]; iy++) {
        auto const py = (iy - sz[1] / 2) * info.voxel_size[1];
        for (long ix = 0; ix < sz[0]; ix++) {
          auto const px = (ix - sz[0] / 2) * info.voxel_size[0];
          Eigen::Vector3f const pos(px, py, pz);
          Eigen::Vector3f const vec = pos - chan_pos;
          float const r = vec.norm() < avoid_div_zero ? 0.f : sense_rad_mm / vec.norm();
          all(ic, ix, iy, iz) = std::polar(r, atan2(vec(0), vec(1)));
        }
      }
    }
  }
  return all;
}

int main_phantom(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "Filename to write phantom data to");
  args::ValueFlag<std::string> suffix(
      parser, "SUFFIX", "Add suffix (well, infix) to output dirs", {"suffix"});
  args::ValueFlag<float> osamp(
      parser, "GRID OVERSAMPLE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::ValueFlag<float> fov(
      parser, "FOV", "Field of View in mm (default 256)", {'f', "fov"}, 240.f);
  args::ValueFlag<long> matrix(parser, "MATRIX", "Matrix size (default 128)", {'m', "matrix"}, 128);
  args::ValueFlag<float> phan_r(
      parser,
      "PHANTOM RADIUS",
      "Radius of the spherical phantom in mm (default 60)",
      {"phan_rad"},
      60.f);
  args::ValueFlag<float> coil_r(
      parser, "COIL RADIUS", "Radius of the coil in mm (default 100)", {"coil_rad"}, 100.f);
  args::ValueFlag<float> oversamp(
      parser, "READ OVERSAMPLE", "Read-out oversampling (default 2)", {'o', "oversamp"}, 2);
  args::ValueFlag<float> spoke_samp(
      parser, "SPOKE SAMPLING", "Sample factor for spokes (default 1)", {'s', "spokes"}, 1);
  args::ValueFlag<long> lores(
      parser, "LO-RES", "Include lo-res k-space with scale factor (suggest 8)", {'l', "lores"}, 0);
  args::ValueFlag<long> gap(parser, "DEAD-TIME", "Samples in dead-time (def 0)", {'g', "gap"}, 0);
  args::ValueFlag<long> nchan(
      parser, "CHANNELS", "Number of channels (default 12)", {'c', "channels"}, 12);
  args::ValueFlag<float> intensity(
      parser, "INTENSITY", "Phantom intensity (default 1000)", {'i', "intensity"}, 1000.f);
  args::ValueFlag<float> snr(parser, "SNR", "Add noise (specified as SNR)", {'n', "snr"}, 0);
  args::Flag kb(parser, "KAISER-BESSEL", "Use Kaiser-Bessel kernel", {'k', "kb"});
  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  parser.Parse();
  Log log(verbose);
  FFT::Start(log);
  if (!fname) {
    log.fail("No output name specified");
  }
  log.info(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());
  auto const m = matrix.Get();
  auto const vox_sz = fov.Get() / m;
  auto const spokes_hi = std::lrint(spoke_samp.Get() * m * m);
  Info info{.matrix = Array3l{m, m, m},
            .read_points = static_cast<long>(m * oversamp.Get() / 2),
            .read_gap = gap.Get(),
            .spokes_hi = spokes_hi,
            .spokes_lo = lores ? static_cast<long>(spokes_hi / lores.Get()) : 0,
            .lo_scale = lores ? lores.Get() : 1.f,
            .channels = nchan.Get(),
            .voxel_size = Eigen::Array3f{vox_sz, vox_sz, vox_sz}};
  log.info(
      FMT_STRING("Matrix Size: {} Voxel Size: {} Oversampling: {} Dead-time Gap: {}"),
      matrix.Get(),
      vox_sz,
      oversamp.Get(),
      gap.Get());
  log.info(FMT_STRING("Hi-res spokes: {} Lo-res spokes: {}"), info.spokes_hi, info.spokes_lo);

  auto traj = ArchimedeanSpiral(info);
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), false) : (Kernel *)new NearestNeighbour();
  Gridder gridder(info, traj, osamp.Get(), false, kernel, false, log);
  Cx4 grid = gridder.newGrid();
  Cropper cropper(info, gridder.gridDims(), -1, false, log);
  Cx3 phan = cropper.newImage();

  // Draw a spherical phantom
  log.info("Drawing phantom...");
  phan.setZero();
  long const cn = phan.dimension(0) / 2;
  long const pr = phan_r.Get();
  for (long iz = cn - pr; iz <= cn + pr; iz++) {
    auto const pz = (iz - cn) * vox_sz;
    for (long iy = cn - pr; iy <= cn + pr; iy++) {
      auto const py = (iy - cn) * vox_sz;
      for (long ix = cn - pr; ix <= cn + pr; ix++) {
        auto const px = (ix - cn) * vox_sz;
        float const rad = sqrt(px * px + py * py + pz * pz);
        if (rad < pr) {
          phan(ix, iy, iz) = intensity.Get();
        }
      }
    }
  }

  // Generate SENSE maps and multiply
  log.info("Generating coil sensitivities...");
  Cx4 sense = birdcage(phan.dimensions(), info.channels, coil_r.Get(), coil_r.Get() / 2.f, info);
  log.info("Generating coil images...");
  cropper.crop4(grid) = sense * Tile(phan, info.channels);
  FFT3N fft(grid, log);
  fft.forward();
  Cx3 radial = info.noncartesianVolume();
  gridder.toNoncartesian(grid, radial);
  if (snr) {
    Cx3 noise(info.read_points, info.spokes_total(), nchan.Get());
    noise.setRandom<Eigen::internal::NormalRandomGenerator<std::complex<float>>>();
    radial.slice(Dims3{0, 0, 0}, Dims3{info.channels, info.read_points, info.spokes_total()}) +=
        noise * noise.constant(intensity.Get() / snr.Get());
  }
  if (gap) {
    radial.slice(Dims3{0, 0, 0}, Dims3{info.channels, gap.Get(), info.spokes_total()}).setZero();
  }

  HD5Writer writer(std::filesystem::path(fname.Get()).replace_extension(".h5").string(), log);
  writer.writeInfo(info);
  writer.writeTrajectory(traj);
  writer.writeData(0, radial);
  FFT::End(log);
  return EXIT_SUCCESS;
}
