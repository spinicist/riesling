#include "kernel.hpp"

#include "kernel-fixed.hpp"
#include "kernel-nn.hpp"

#include "expsemi.hpp"
#include "kaiser.hpp"

#include "../tensors.hpp"

#include <fmt/ranges.h>

namespace rl {

template <typename Scalar, int ND, typename Func> struct Kernel final : KernelBase<Scalar, ND>
{
  static constexpr int Width = Func::Width;
  static constexpr int PadWidth = (((Width + 1) / 2) * 2) + 1;
  using Tensor = typename FixedKernel<float, ND, Func>::Tensor;
  using Point = typename FixedKernel<float, ND, Func>::Point;

  Func  f;
  float scale;

  Kernel(float const osamp)
    : f(osamp)
    , scale{1.f}
  {
    static_assert(ND < 4);
    scale = 1. / Norm(FixedKernel<Scalar, ND, Func>::Kernel(f, 1.f, Point::Zero()));
    Log::Print("Kernel", "Width {} Scale {}", Func::Width, scale);
  }

  virtual auto paddedWidth() const -> int final { return PadWidth; }

  inline auto operator()(Point const p) const -> Eigen::Tensor<float, ND> final
  {
    Tensor                   k = FixedKernel<Scalar, ND, Func>::Kernel(f, scale, p);
    Eigen::Tensor<float, ND> k2 = k;
    return k;
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, y, x);
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &b,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, b, y, x);
  }

  void gather(Eigen::Array<int16_t, ND, 1> const                           c,
              Point const                                                 &p,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, x, y);
  }

  void gather(Eigen::Array<int16_t, ND, 1> const                           c,
              Point const                                                 &p,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, b, x, y);
  }
};

template <typename Scalar, int ND>
auto KernelBase<Scalar, ND>::Make(std::string const &kType, float const osamp) -> std::shared_ptr<KernelBase<Scalar, ND>>
{
  if (kType == "NN") {
    return std::make_shared<NearestNeighbour<Scalar, ND>>();
  } else if (kType.size() == 3) {
    std::string const type = kType.substr(0, 2);
    int const         W = std::stoi(kType.substr(2, 1));
    if (type == "ES") {
      switch (W) {
      case 2: return std::make_shared<Kernel<Scalar, ND, ExpSemi<2>>>(osamp);
      case 4: return std::make_shared<Kernel<Scalar, ND, ExpSemi<4>>>(osamp);
      case 6: return std::make_shared<Kernel<Scalar, ND, ExpSemi<6>>>(osamp);
      default: throw Log::Failure("Kernel", "Unsupported width {}", W);
      }
    } else if (type == "KB") {
      switch (W) {
      case 2: return std::make_shared<Kernel<Scalar, ND, ExpSemi<2>>>(osamp);
      case 4: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<4>>>(osamp);
      case 6: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<6>>>(osamp);
      default: throw Log::Failure("Kernel", "Unsupported width {}", W);
      }
    }
  }

  throw Log::Failure("Kernel", "Unknown type {}", kType);
}

} // namespace rl