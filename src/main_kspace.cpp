#include "types.h"

#include "fft.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include <filesystem>

int main_kspace(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "Input k-space file to take images of");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<long> volume(parser, "VOLUME", "Output this volume (default first)", {"vol"}, 0);
  args::ValueFlag<long> start(parser, "START", "Start index for spokes, default 0", {"start"}, 0);
  args::ValueFlag<long> spokes(
      parser, "SPOKES", "Number of spokes, default 128", {'s', "spokes"}, 128);
  args::ValueFlag<long> samples(
      parser,
      "SAMPLES",
      "Number of radial/gridded samples to write out, default all",
      {"samples"},
      -1);
  args::ValueFlag<float> osamp(parser, "OSAMP", "Grid oversampling factor (2)", {'s', "os"}, 2.f);
  args::Flag est_dc(parser, "ESTIMATE DC", "Estimate DC weights instead of analytic", {"est_dc"});
  args::Flag kb(parser, "KB", "Use Kaiser-Bessel interpolation", {"kb"});
  args::Flag stack(parser, "STACK", "Trajectory is stack-of-stars or similar", {"stack"});
  args::Flag do_grid(parser, "GRID", "Grid k-space and write out central portion", {"grid"});
  args::Flag do_regrid(parser, "REGRID", "Grid back to radial and write out", {"regrid"});
  args::ValueFlag<float> scale(
      parser, "SCALE", "Scale output by this factor (default full-range)", {"scale"}, -1.f);
  args::Flag lores(parser, "LO-RES", "Write out lo-res spokes, not hi-res", {'l', "lores"});
  Log log = ParseCommand(parser, fname);
  FFTStart(log);
  HD5Reader reader(fname.Get(), log);
  auto info = reader.info();

  Cx3 rad_ks = info.noncartesianVolume();
  reader.readData(volume.Get(), rad_ks);

  long const n_samp = samples ? samples.Get() : info.read_points;
  Dims3 const st{0, 0, start.Get() + (lores ? 0 : info.spokes_lo)};
  Dims3 sz{info.channels, n_samp, spokes.Get()};
  if (!do_grid || !do_regrid) {
    WriteNifti(info, Cx3(rad_ks.slice(st, sz)), OutName(fname, oname, "kspace-radial"), log);
  }

  if (do_grid || do_regrid) {
    auto const res = info.voxel_size.minCoeff() * info.read_points / n_samp;
    log.info(FMT_STRING("Gridding with {} samples, nominal resolution {} mm"), n_samp, res);
    R3 const traj = reader.readTrajectory();
    Gridder gridder(info, traj, osamp.Get(), est_dc, kb, stack, log, res, true);
    Cx4 grid = gridder.newGrid();
    FFT3N fft(grid, log);
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.shift();
    if (do_grid) {
      WriteNifti(
          info, Cx4(grid.shuffle(Sz4{1, 2, 3, 0})), OutName(fname, oname, "kspace-cartesian"), log);
    }
    if (do_regrid) {
      R3 traj2 = traj.slice(Sz3{0, 0, info.spokes_lo}, Sz3{3, info.read_points, info.spokes_hi});
      info.spokes_lo = 0;
      Gridder regridder(info, traj2, osamp.Get(), est_dc, kb, stack, log, res, true);
      info.read_gap = 0;
      info.spokes_lo = 0;
      fft.shift();
      rad_ks.setZero();
      regridder.toNoncartesian(grid, rad_ks);
      WriteNifti(info, Cx3(rad_ks.slice(st, sz)), OutName(fname, oname, "kspace-radial"), log);
    }
  }
  FFTEnd(log);
  log.info("Finished");
  return EXIT_SUCCESS;
}
