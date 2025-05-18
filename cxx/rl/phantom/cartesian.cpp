#include "cartesian.hpp"
#include "../log/log.hpp"

namespace rl {

Re3 Cartesian(Index const M)
{
  Index const nTrace = M * M;
  Index const nRead = M;

  Re3 traj(3, nRead, nTrace);

  for (Index iz = 0; iz < M; iz++) {
    float const rz = iz - M / 2;
    for (Index iy = 0; iy < M; iy++) {
      float const ry = iy - M / 2;
      for (Index ix = 0; ix < M; ix++) {
        float const rx = ix - M / 2;
        traj(0, ix, iy + iz * M) = rx;
        traj(1, ix, iy + iz * M) = ry;
        traj(2, ix, iy + iz * M) = rz;
      }
    }
  }

  return traj;
}
} // namespace rl
