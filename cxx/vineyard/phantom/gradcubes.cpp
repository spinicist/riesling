#include "gradcubes.hpp"

#include "log.hpp"

namespace rl {

/* Gradient cubes for TGV */

Cx3 GradCubes(Sz3 const &matrix, Eigen::Array3f const &voxel_size, float const hsz)
{
  Log::Print("Phan", "Drawing Gradient Cubes");
  Cx3 phan(matrix[0], matrix[1], matrix[2]);
  phan.setZero();

  Index const cx = phan.dimension(0) / 2;
  Index const cy = phan.dimension(1) / 2;
  Index const cz = phan.dimension(2) / 2;

  // Three cubes, one for each axis

  for (Index iz = 0; iz < phan.dimension(2); iz++) {
    auto const pz = (iz - cz) * voxel_size[2];
    for (Index iy = 0; iy < phan.dimension(1); iy++) {
      auto const py = (iy - cy) * voxel_size[1];
      for (Index ix = 0; ix < phan.dimension(0); ix++) {
        auto const           px = (ix - cx) * voxel_size[0];
        Eigen::Array3f const p{px, py, pz};
        Eigen::Array3f const r = p / hsz; // Normalize coordinates between -1 and 1
        if ((r.abs() <= 1.f / 3.f).all()) {
          phan(ix, iy, iz) = r(0) * 3.f + 2.f;
        } else if ((r.abs() <= 2.f / 3.f).all()) {
          phan(ix, iy, iz) = r(1) * 3.f / 2.f + 2.f;
        } else if ((r.abs() <= 1.f).all()) {
          phan(ix, iy, iz) = r(2) + 2.f;
        }
      }
    }
  }
  return phan;
}
} // namespace rl
