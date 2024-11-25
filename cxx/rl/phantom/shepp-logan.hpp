#pragma once

#include "../types.hpp"

namespace rl {

Cx3 SheppLoganPhantom(
  Sz3 const                          &matrix,
  Eigen::Array3f const               &voxel_size,
  Eigen::Vector3f const              &c,
  Eigen::Vector3f const              &imr,
  float const                         rad,
  std::vector<Eigen::Vector3f> const &centres,
  std::vector<Eigen::Array3f> const  &ha,
  std::vector<float> const           &angles,
  std::vector<float> const           &intensities);

}
