#include "shepp-logan.hpp"

#include "../log.hpp"

namespace rl {

/* Creates a 3D Shepp Logan using the method outlined in
 * Cheng, G. K., Sarlls, J. E., & Özarslan, E. (2007).
 * Three-dimensional analytical magnetic resonance imaging phantom in the Fourier domain.
 * Magnetic Resonance in Medicine, 58(2), 430–436. https://doi.org/10.1002/mrm.21292
 *
 * Here we calculate the phantom in image space instead of k-space though to use with sense maps
 */

Cx3 SheppLoganPhantom(
  Sz3 const                          &matrix,
  Eigen::Array3f const               &voxel_size,
  Eigen::Vector3f const              &c,
  Eigen::Vector3f const              &imr,
  float const                         rad,
  std::vector<Eigen::Vector3f> const &centres,
  std::vector<Eigen::Array3f> const  &ha,
  std::vector<float> const           &angles,
  std::vector<float> const           &intensities)
{
  if (!(centres.size() == ha.size() && centres.size() == angles.size() && centres.size() == intensities.size())) {
    throw Log::Failure("Phan", "Shepp Logan property lengths did not match");
  }

  Log::Print("Phan", "Drawing 3D Shepp Logan");
  Cx3 phan(matrix[0], matrix[1], matrix[2]);
  phan.setZero();

  Index const cx = phan.dimension(0) / 2;
  Index const cy = phan.dimension(1) / 2;
  Index const cz = phan.dimension(2) / 2;

  // Global rotation
  Eigen::Matrix3f yaw;
  Eigen::Matrix3f pitch;
  Eigen::Matrix3f roll;
  float           d2r = M_PI / 180.0;
  roll << 1.f, 0.f, 0.f,                        //
    0.f, cos(imr[0] * d2r), -sin(imr[0] * d2r), //
    0.f, sin(imr[0] * d2r), cos(imr[0] * d2r);

  pitch << cos(imr[1] * d2r), 0.f, sin(imr[1] * d2r), //
    0.f, 1.f, 0.f,                                    //
    -sin(imr[1] * d2r), 0.f, cos(imr[1] * d2r);

  yaw << cos(imr[2] * d2r), -sin(imr[2] * d2r), 0.f, //
    sin(imr[2] * d2r), cos(imr[2] * d2r), 0.f,       //
    0.f, 0.f, 1.f;

  for (Index iz = 0; iz < phan.dimension(2); iz++) {
    auto const pz = (iz - cz) * voxel_size[2];
    for (Index iy = 0; iy < phan.dimension(1); iy++) {
      auto const py = (iy - cy) * voxel_size[1];
      for (Index ix = 0; ix < phan.dimension(0); ix++) {
        auto const            px = (ix - cx) * voxel_size[0];
        Eigen::Vector3f       p0{px, py, pz};
        Eigen::Vector3f const p = yaw * pitch * roll * p0;

        // Normalize coordinates between -1 and 1
        Eigen::Vector3f const r = (p - c) / rad;

        // Loop over the 10 elipsoids
        for (Index ie = 0; ie < 10; ie++) {
          Eigen::Matrix3f rot;
          rot << cos(angles[ie]), sin(angles[ie]), 0.f, //
            -sin(angles[ie]), cos(angles[ie]), 0.f,     //
            0.f, 0.f, 1.f;

          Eigen::Vector3f const pe = ((rot * (r - centres[ie])).array() / ha[ie]);
          if (pe.norm() < 1.f) { phan(ix, iy, iz) += intensities[ie]; }
        }
      }
    }
  }
  return phan;
}
} // namespace rl
