#pragma once

#include "types.hpp"

namespace rl {

template <typename Scalar, int ND> struct KernelBase
{
  using Ptr = std::shared_ptr<KernelBase>;
  using Point = Eigen::Matrix<float, ND, 1>;
  virtual auto paddedWidth() const -> int = 0;
  virtual auto operator()(Point const p) const -> Eigen::Tensor<float, ND> = 0;
  virtual void spread(Eigen::Array<int16_t, ND, 1> const c,
                      Point const                       &p,
                      Eigen::Tensor<Scalar, 1> const    &y,
                      Eigen::Tensor<Scalar, ND + 2>     &x) const = 0;
  virtual void spread(Eigen::Array<int16_t, ND, 1> const c,
                      Point const                       &p,
                      Eigen::Tensor<Scalar, 1> const    &b,
                      Eigen::Tensor<Scalar, 1> const    &y,
                      Eigen::Tensor<Scalar, ND + 2>     &x) const = 0;
  virtual void gather(Eigen::Array<int16_t, ND, 1> const                           c,
                      Point const                                                 &p,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const = 0;
  virtual void gather(Eigen::Array<int16_t, ND, 1> const                           c,
                      Point const                                                 &p,
                      Eigen::Tensor<Scalar, 1> const                              &b,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                      Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const = 0;

  static auto Make(std::string const &type, float const osamp) -> std::shared_ptr<KernelBase<Scalar, ND>>;
};

} // namespace rl
