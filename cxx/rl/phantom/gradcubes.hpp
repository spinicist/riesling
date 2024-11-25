#pragma once

#include "../types.hpp"

namespace rl {

Cx3 GradCubes(Sz3 const &matrix, Eigen::Array3f const &voxel_size, float const hsz);

}
