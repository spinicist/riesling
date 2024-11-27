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
  using Point = typename FixedKernel<float, ND, Func>::Point;
  template <int TND> using Tensor = Eigen::Tensor<Scalar, TND>;
  template <int TND> using MMap = Eigen::TensorMap<Eigen::Tensor<Scalar, TND>>;
  template <int TND> using CMap = Eigen::TensorMap<Eigen::Tensor<Scalar, TND> const>;

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
    return FixedKernel<Scalar, ND, Func>::Kernel(f, scale, p);
  }

  void spread(
    Eigen::Array<int16_t, ND, 1> const c, Point const &p, Scalar const w, Tensor<1> const &y, Tensor<ND + 2> &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, c, p, scale * w, y, x);
  }

  void spread(
    Eigen::Array<int16_t, ND, 1> const c, Point const &p, Tensor<1> const &b, Tensor<1> const &y, Tensor<ND + 2> &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, c, p, scale * b, y, x);
  }

  void
  gather(Eigen::Array<int16_t, ND, 1> const c, Point const &p, Scalar const w, CMap<ND + 2> const &x, MMap<1> &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, c, p, scale * w, x, y);
  }

  void gather(
    Eigen::Array<int16_t, ND, 1> const c, Point const &p, Tensor<1> const &b, CMap<ND + 2> const &x, MMap<1> &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, c, p, scale * b, x, y);
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