#include "phantom_sphere.h"

Cx3 SphericalPhantom(
    Array3l const &matrix,
    Eigen::Array3f const &voxel_size,
    Eigen::Vector3f const &c,
    float const r,
    float const i,
    Log const &log)
{

  // Draw a spherical phantom
  log.info(FMT_STRING("Drawing sphere center {} radius {} mm intensity {}"), c.transpose(), r, i);
  Cx3 phan(matrix[0], matrix[1], matrix[2]);
  phan.setZero();
  long const cx = phan.dimension(0) / 2;
  long const cy = phan.dimension(1) / 2;
  long const cz = phan.dimension(2) / 2;

  for (long iz = 0; iz < phan.dimension(2); iz++) {
    auto const pz = (iz - cz) * voxel_size[2];
    for (long iy = 0; iy < phan.dimension(1); iy++) {
      auto const py = (iy - cy) * voxel_size[1];
      for (long ix = 0; ix < phan.dimension(0); ix++) {
        auto const px = (ix - cx) * voxel_size[0];
        Eigen::Vector3f const p{px, py, pz};
        if ((p - c).norm() < r) {
          phan(ix, iy, iz) = i;
        }
      }
    }
  }

  log.image(phan, "phantom-sphere.nii");
  return phan;
}
