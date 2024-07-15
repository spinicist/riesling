#pragma once

#include "types.hpp"

namespace rl {

template <typename Scalar, int ND> struct Kernel
{
  using Ptr = std::shared_ptr<Kernel>;
  using Point = Eigen::Matrix<float, ND, 1>;
  virtual auto paddedWidth() const -> int = 0;
  virtual auto at(Point const p) const -> Eigen::Tensor<float, ND> = 0;
  virtual void spread(std::array<int16_t, ND> const   c,
                      Point const                     p,
                      Eigen::Tensor<Scalar, 1> const &b,
                      Eigen::Tensor<Scalar, 1> const &y,
                      Eigen::Tensor<Scalar, ND + 2>  &x) const = 0;

  virtual void gather(std::array<int16_t, ND> const                                c,
                      Point const                                                  p,
                      Eigen::Tensor<Scalar, 1> const                              &b,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const = 0;

  static auto Make(std::string const &type, float const osamp) -> std::shared_ptr<Kernel<Scalar, ND>>;
};

} // namespace rl
