#pragma once

#include "../types.hpp"

namespace rl {

template <typename Scalar, int ND> struct KernelBase
{
  using Ptr = std::shared_ptr<KernelBase>;
  using Point = Eigen::Matrix<float, ND, 1>;
  template <int TND> using Tensor = Eigen::Tensor<Scalar, TND>;
  template <int TND> using MMap = Eigen::TensorMap<Eigen::Tensor<Scalar, TND>>;
  template <int TND> using CMap = Eigen::TensorMap<Eigen::Tensor<Scalar, TND> const>;
  virtual auto paddedWidth() const -> int = 0;
  virtual auto operator()(Point const p) const -> Eigen::Tensor<float, ND> = 0;
  virtual void
  spread(Eigen::Array<int16_t, ND, 1> const c, Point const &p, Scalar const w, Tensor<1> const &y, Tensor<ND + 2> &x) const = 0;
  virtual void spread(
    Eigen::Array<int16_t, ND, 1> const c, Point const &p, Tensor<1> const &b, Tensor<1> const &y, Tensor<ND + 2> &x) const = 0;
  virtual void
  gather(Eigen::Array<int16_t, ND, 1> const c, Point const &p, Scalar const w, CMap<ND + 2> const &x, MMap<1> &y) const = 0;
  virtual void
  gather(Eigen::Array<int16_t, ND, 1> const c, Point const &p, Tensor<1> const &b, CMap<ND + 2> const &x, MMap<1> &y) const = 0;

  static auto Make(std::string const &type, float const osamp) -> std::shared_ptr<KernelBase<Scalar, ND>>;
};

} // namespace rl
