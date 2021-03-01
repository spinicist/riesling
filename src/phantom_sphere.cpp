#include "phantom_sphere.h"

Cx3 SphericalPhantom(Info const &info, float const radius, float const intensity, Log const &log)
{

  // Draw a spherical phantom
  log.info(FMT_STRING("Drawing spherical phantom radius {} mm intensity {}"), radius, intensity);
  Cx3 phan(info.matrix[0], info.matrix[1], info.matrix[2]);
  phan.setZero();
  long const cx = phan.dimension(0) / 2;
  long const cy = phan.dimension(1) / 2;
  long const cz = phan.dimension(2) / 2;

  for (long iz = 0; iz < phan.dimension(2); iz++) {
    auto const pz = (iz - cz) * info.voxel_size[2];
    for (long iy = 0; iy < phan.dimension(1); iy++) {
      auto const py = (iy - cy) * info.voxel_size[1];
      for (long ix = 0; ix < phan.dimension(0); ix++) {
        auto const px = (ix - cx) * info.voxel_size[0];
        float const rad = sqrt(px * px + py * py + pz * pz);
        if (rad < radius) {
          phan(ix, iy, iz) = intensity;
        }
      }
    }
  }

  log.image(phan, "phantom-sphere.nii");
  return phan;
}