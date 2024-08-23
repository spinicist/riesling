#pragma once

#include "log.hpp"
#include "types.hpp"

namespace rl {

/* This function is ported from ismrmd_phantom.cpp, which in turn is inspired by
 * http://web.eecs.umich.edu/~fessler/code/
 */
auto birdcage(Sz3 const            &matrix,
              Eigen::Array3f const &voxel_size,
              Index const           channels,
              Index const           nrings,
              float const           coil_rad_mm, // Radius of the actual coil, i.e. where the channels should go
              float const           sense_rad_mm) -> Cx4;
} // namespace rl
