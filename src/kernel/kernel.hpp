#pragma once

#include "types.hpp"

namespace rl {

template <typename Scalar, int ND>
struct Kernel
{
  using Point = Eigen::Matrix<float, ND, 1>;
  virtual auto paddedWidth() const -> Index = 0;
  virtual void setOversampling(float const os) = 0;
  virtual auto at(Point const p) const -> Eigen::Tensor<float, ND> = 0;
  virtual void spread(std::array<int16_t, ND> const   c,
                      Point const                     p,
                      Sz<ND> const                    minCorner,
                      Eigen::Tensor<float, 1> const  &b,
                      Eigen::Tensor<Scalar, 1> const &y,
                      Eigen::Tensor<Scalar, ND + 2>  &x) const = 0;

  virtual auto gather(std::array<int16_t, ND> const                                c,
                      Point const                                                  p,
                      Sz<ND> const                                                 minCorner,
                      Eigen::Tensor<float, 1> const                               &b,
                      Sz<ND> const                                                 cdims,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x) const -> Eigen::Tensor<Scalar, 1> = 0;
};

template <typename Scalar, int ND>
auto make_kernel(std::string const &type, float const osamp) -> std::shared_ptr<Kernel<Scalar, ND>>;

} // namespace rl
