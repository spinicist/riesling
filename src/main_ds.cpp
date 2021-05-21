#include "types.h"

#include "filter.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"

using namespace std::complex_literals;

constexpr float pi = M_PI;

int main_ds(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "Input radial k-space file");

  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {"out", 'o'});
  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  Log log = ParseCommand(parser, fname);
  HD5::Reader reader(fname.Get(), log);
  Trajectory traj = reader.readTrajectory();
  auto const &info = traj.info();

  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 channels(info.channels, info.matrix[0], info.matrix[1], info.matrix[2]);
  R4 out(info.matrix[0], info.matrix[1], info.matrix[2], info.volumes);
  auto const sx = info.matrix[0];
  auto const hx = sx / 2;
  auto const sy = info.matrix[1];
  auto const hy = sy / 2;
  auto const sz = info.matrix[2];
  auto const hz = sz / 2;
  auto const maxX = info.matrix.maxCoeff();
  auto const maxK = maxX / 2;

  float const scale = std::sqrt(info.matrix.prod());

  // Work out volume element
  auto const delta = 1.;
  float const d_lo = (4. / 3.) * M_PI * delta * delta * delta / info.spokes_lo;
  float const d_hi = (4. / 3.) * M_PI * delta * delta * delta / info.spokes_hi;

  // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
  float const approx_undersamp =
      (M_PI * info.matrix.maxCoeff() * info.matrix.maxCoeff()) / info.spokes_hi;
  float const flat_start = maxK / sqrt(approx_undersamp);
  float const flat_val = d_hi * (3. * (flat_start * flat_start) + 1. / 4.);

  auto const &all_start = log.now();
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const &vol_start = log.now();
    reader.readNoncartesian(iv, rad_ks);
    channels.setZero();
    log.info("Beginning Direct Summation");
    auto fourier = [&](long const lo, long const hi) {
      for (long iz = lo; iz < hi; iz++) {
        log.info("Starting {}/{}", iz, hi);
        for (long iy = 0; iy < sy; iy++) {
          for (long ix = 0; ix < sx; ix++) {
            Point3 const c = Point3{
                (ix - hx) / (float)(maxX), (iy - hy) / (float)(maxX), (iz - hz) / (float)(maxX)};
            for (long is = 0; is < info.spokes_total(); is++) {
              for (long ir = info.read_gap; ir < info.read_points; ir++) {
                Point3 const r = traj.point(ir, is, maxK);
                float const r_mag = r.matrix().norm();
                auto const &d_k = is < info.spokes_lo ? d_lo : d_hi;
                float dc;
                if (r_mag == 0.f) {
                  dc = d_k * 1.f / 8.f;
                } else if (r_mag < flat_start) {
                  dc = d_k * (3. * (r_mag * r_mag) + 1. / 4.);
                } else {
                  dc = flat_val;
                }
                std::complex<float> const f_term =
                    std::exp(2.if * pi * r.matrix().dot(c.matrix())) * dc / scale;

                for (long ic = 0; ic < info.channels; ic++) {
                  auto const val = rad_ks(ic, ir, is) * f_term;
                  channels(ic, ix, iy, iz) += val;
                }
              }
            }
          }
        }
        log.info("Finished {}/{}", iz, hi);
      }
    };
    Threads::RangeFor(fourier, sz);
    log.info("Calculating RSS");
    WriteNifti(info, Cx4(channels.shuffle(Sz4{1, 2, 3, 0})), "chan.nii", log);
    out.chip(iv, 3).device(Threads::GlobalDevice()) =
        (channels * channels.conjugate()).sum(Sz1{0}).real().sqrt();
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All volumes: {}", log.toNow(all_start));

  WriteVolumes(info, R4(out.abs()), volume.Get(), OutName(fname, oname, "ds"), log);
  return EXIT_SUCCESS;
}
