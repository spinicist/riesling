#pragma once

namespace rl {

template <int ND> struct GridOpts
{
  using Arrayf = Eigen::Array<float, ND, 1>;
  Arrayf fov = Arrayf::Zero();
  float  osamp = 1.3f;
  bool tophat = false;
  Index kW = 4;
};

}
