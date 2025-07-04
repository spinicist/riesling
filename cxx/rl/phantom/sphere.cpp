#include "sphere.hpp"

#include "../log/log.hpp"

namespace rl {

Cx3 SphericalPhantom(
  Sz3 const &matrix, Eigen::Array3f const &voxel_size, Eigen::Vector3f const &c, float const r, float const i, bool const hemi)
{

  // Draw a spherical phantom
  Log::Print("Phan", "Drawing {}sphere center {} radius {} mm intensity {}", hemi ? "hemi" : "", c.transpose(), r, i);
  Cx3 phan(matrix[0], matrix[1], matrix[2]);
  phan.setZero();
  Index const cx = phan.dimension(0) / 2;
  Index const cy = phan.dimension(1) / 2;
  Index const cz = phan.dimension(2) / 2;

  Index const stopz = hemi ? phan.dimension(2) / 2 : phan.dimension(2);

  for (Index iz = 0; iz < stopz; iz++) {
    auto const pz = (iz - cz) * voxel_size[2];
    for (Index iy = 0; iy < phan.dimension(1); iy++) {
      auto const py = (iy - cy) * voxel_size[1];
      for (Index ix = 0; ix < phan.dimension(0); ix++) {
        auto const            px = (ix - cx) * voxel_size[0];
        Eigen::Vector3f const p{px, py, pz};
        if ((p - c).norm() < r) { phan(ix, iy, iz) = i; }
      }
    }
  }
  return phan;
}
} // namespace rl
