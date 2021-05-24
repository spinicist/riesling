#include "phantom_shepplogan.h"
#include <stdio.h>

/* Creates a 3D Shepp Logan using the method outlined in
 * Cheng, G. K., Sarlls, J. E., & Özarslan, E. (2007).
 * Three-dimensional analytical magnetic resonance imaging phantom in the Fourier domain.
 * Magnetic Resonance in Medicine, 58(2), 430–436. https://doi.org/10.1002/mrm.21292
 *
 * Here we calculate the phantom in image space instead of k-space though to use with sense maps
 */

namespace {
// Parameters for the 10 elipsoids in the 3D Shepp-Logan phantom from Cheng et al.
std::vector<Eigen::Vector3f> const centres{{0, 0, 0},
                                           {0, 0, 0},
                                           {-0.22, 0, -0.25},
                                           {0.22, 0, -0.25},
                                           {0, 0.35, -0.25},
                                           {0, 0.1, -0.25},
                                           {-0.08, -0.65, -0.25},
                                           {0.06, -0.65, -0.25},
                                           {0.06, -0.105, 0.625},
                                           {0, 0.1, 0.625}};

// Half-axes
std::vector<Eigen::Array3f> const ha{{0.69, 0.92, 0.9},
                                     {0.6624, 0.874, 0.88},
                                     {0.41, 0.16, 0.21},
                                     {0.31, 0.11, 0.22},
                                     {0.21, 0.25, 0.5},
                                     {0.046, 0.046, 0.046},
                                     {0.046, 0.023, 0.02},
                                     {0.046, 0.023, 0.02},
                                     {0.056, 0.04, 0.1},
                                     {0.056, 0.056, 0.1}};

std::vector<float> const angles{0, 0, 3 * M_PI / 5, 2 * M_PI / 5, 0, 0, 0, M_PI / 2, M_PI / 2, 0};
std::vector<float> const pd{2, -0.8, -0.2, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, -0.2};
} // namespace

Cx3 SheppLoganPhantom(
    Array3l const &matrix,
    Eigen::Array3f const &voxel_size,
    Eigen::Vector3f const &c,
    float const rad,
    float const intensity,
    Log const &log)
{
  log.info(FMT_STRING("Drawing 3D Shepp Logan Phantom. Intensity {}"), intensity);
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
        // Normalize coordinates between -1 and 1
        Eigen::Vector3f const r = (p - c) / rad;

        // Loop over the 10 elipsoids
        for (long ie = 0; ie < 10; ie++) {
          Eigen::Matrix3f rot;
          rot << cos(angles[ie]), sin(angles[ie]), 0.f, //
              -sin(angles[ie]), cos(angles[ie]), 0.f,   //
              0.f, 0.f, 1.f;
          Eigen::Vector3f const pe = (rot * (r - centres[ie])).array() / ha[ie];
          if (pe.norm() < 1.f) {
            phan(ix, iy, iz) += pd[ie] * intensity;
          }
        }
      }
    }
  }

  log.image(phan, "phantom-shepplogan.nii");
  return phan;
}