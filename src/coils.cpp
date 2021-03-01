#include "coils.h"

Cx4 birdcage(
    Dims3 const sz,
    long const nchan,
    float const coil_rad_mm,  // Radius of the actual coil, i.e. where the channels should go
    float const sense_rad_mm, // Sensitivity radius
    Info const &info,
    Log const &log)
{
  log.info(FMT_STRING("Constructing bird-cage sensitivities with {} channels"), nchan);
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
  if (log.level() >= Log::Level::Images) { // Extra check to avoid the shuffle when we can
    log.image(SwapToChannelLast(all), fmt::format(FMT_STRING("birdcage-{}.nii"), nchan));
  }
  return all;
}