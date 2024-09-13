#include "kernel.hpp"

#include "kernel-fixed.hpp"
#include "kernel-nn.hpp"

#include "expsemi.hpp"
#include "kaiser.hpp"

#include "tensors.hpp"

#include <fmt/ranges.h>

namespace rl {

template <typename Scalar, int ND, typename Func> struct Kernel final : KernelBase<Scalar, ND>
{
  static constexpr int   Width = Func::Width;
  static constexpr int   PadWidth = Func::PadWidth;
  static constexpr float HalfWidth = Width / 2.f;
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<Func::PadWidth>>;
  using Tensor = typename FixedKernel<float, ND, Func>::Tensor;
  using Point = typename FixedKernel<float, ND, Func>::Point;
  using Pos = typename FixedKernel<float, ND, Func>::OneD;

  Func  f;
  float β, scale;
  OneD  centers;

  Kernel(float const osamp)
    : f(osamp)
    , scale{1.f}
  {
    static_assert(ND < 4);
    for (int ii = 0; ii < Func::PadWidth; ii++) {
      this->centers(ii) = ii + 0.5f - (Func::PadWidth / 2.f);
    }
    scale = 1. / Norm(FixedKernel<Scalar, ND, Func>::K(f, 1.f, Point::Zero()));
  Log::Print("Kernel", "Scale {}", scale);
  }

  virtual auto paddedWidth() const -> int final { return Func::PadWidth; }

  inline auto operator()(Point const p) const -> Eigen::Tensor<float, ND> final
  {
    Tensor                   k = FixedKernel<Scalar, ND, Func>::K(f, scale, p);
    Eigen::Tensor<float, ND> k2 = k;
    return k;
  }

  void spread(std::array<int16_t, ND> const   c,
              Point const                    &p,
              Eigen::Tensor<Scalar, 1> const &y,
              Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, y, x);
  }

  void spread(std::array<int16_t, ND> const   c,
              Point const                    &p,
              Eigen::Tensor<Scalar, 1> const &b,
              Eigen::Tensor<Scalar, 1> const &y,
              Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, b, y, x);
  }

  void gather(std::array<int16_t, ND> const                                c,
              Point const                                                 &p,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, x, y);
  }

  void gather(std::array<int16_t, ND> const                                c,
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
      case 3: return std::make_shared<Kernel<Scalar, ND, ExpSemi<3>>>(osamp);
      case 4: return std::make_shared<Kernel<Scalar, ND, ExpSemi<4>>>(osamp);
      case 5: return std::make_shared<Kernel<Scalar, ND, ExpSemi<5>>>(osamp);
      case 7: return std::make_shared<Kernel<Scalar, ND, ExpSemi<7>>>(osamp);
      default: throw Log::Failure("Kernel", "Unsupported width {}", W);
      }
    } else if (type == "KB") {
      switch (W) {
      case 3: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<3>>>(osamp);
      case 4: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<4>>>(osamp);
      case 5: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<5>>>(osamp);
      case 7: return std::make_shared<Kernel<Scalar, ND, KaiserBessel<7>>>(osamp);
      default: throw Log::Failure("Kernel", "Unsupported width {}", W);
      }
    }
  }

  throw Log::Failure("Kernel", "Unknown type {}", kType);
}

} // namespace rl