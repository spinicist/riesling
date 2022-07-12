#include "coils.h"

namespace rl {

Cx4 birdcage(
  Eigen::Array3l const &matrix,
  Eigen::Array3f const &voxel_size,
  Index const channels,
  Index const nrings,
  float const coil_rad_mm, // Radius of the actual coil, i.e. where the channels should go
  float const sense_rad_mm)
{
  Log::Print(FMT_STRING("Constructing bird-cage sensitivities with {} channels in {} rings"), channels, nrings);
  Cx4 all(channels, matrix[0], matrix[1], matrix[2]);

  if (channels < 1) {
    Log::Fail(FMT_STRING("Must have at least one channel for birdcage sensitivities"));
  } else if (channels == 1) {
    all.setConstant(1.f);
  } else {
    auto const avoid_div_zero = voxel_size.matrix().norm() / 2.f;
    Index const chan_per_ring = channels / nrings;
    float const fovz = (voxel_size[2] * matrix[2]);
    for (Index ir = 0; ir < nrings; ir++) {
      float const cz = fovz * 2.f * (-0.5f + (ir + 1.f) / (nrings + 1.f));
      for (Index ic = 0; ic < chan_per_ring; ic++) {
        float const theta = (2.f * M_PI * ic) / chan_per_ring;
        float const coil_phs = -(2.f * M_PI * (ic + (ir * chan_per_ring))) / channels;
        Eigen::Vector3f const chan_pos =
          Eigen::Vector3f(coil_rad_mm * std::cos(theta), coil_rad_mm * std::sin(theta), cz);
        for (Index iz = 0; iz < matrix[2]; iz++) {
          auto const pz = (iz - matrix[2] / 2) * voxel_size[2];
          for (Index iy = 0; iy < matrix[1]; iy++) {
            auto const py = (iy - matrix[1] / 2) * voxel_size[1];
            for (Index ix = 0; ix < matrix[0]; ix++) {
              auto const px = (ix - matrix[0] / 2) * voxel_size[0];
              Eigen::Vector3f const pos(px, py, pz);
              Eigen::Vector3f const vec = pos - chan_pos;
              float const r = vec.norm() < avoid_div_zero ? 0.f : sense_rad_mm / vec.norm();
              all(ic + ir * chan_per_ring, ix, iy, iz) = std::polar(r, atan2(vec(0), vec(1)) + coil_phs);
            }
          }
        }
      }
    }
  }
  Log::Tensor(all, fmt::format(FMT_STRING("birdcage-{}"), channels));
  return all;
}
} // namespace rl
