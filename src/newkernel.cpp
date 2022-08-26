#include "newkernel.hpp"

#include "log.h"

namespace rl {

template <size_t W>
struct NewSizedKernel<1, W> : NewKernel
{
  using KTensor = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::declval<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W + 2)...>>>();
  }(std::make_index_sequence<1>()));
  using KPoint = Eigen::Matrix<float, 1, 1>;

  auto distSq(KPoint const p) const -> KTensor
  {
    // Yes, below is a weird mix of integer and floating point division.
    // But it works
    KTensor k;
    constexpr float W_2 = W / 2.f;
    constexpr Index IW_2 = (W + 2) / 2;

    for (Index ix = 0; ix < (W + 2); ix++) {
      float const dx = (IW_2 + p(0) - ix) / W_2;
      float const dx2 = dx * dx;
      k(ix) = (dx2);
    }

    return k;
  }
};

template <size_t W>
struct NewSizedKernel<2, W> : NewKernel
{
  using KTensor = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::declval<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W + 2)...>>>();
  }(std::make_index_sequence<2>()));
  using KPoint = Eigen::Matrix<float, 2, 1>;

  auto distSq(KPoint const p) const -> KTensor
  {
    KTensor k;
    constexpr float W_2 = W / 2.f;
    constexpr Index IW_2 = (W + 2) / 2;

    for (Index iy = 0; iy < (W + 2); iy++) {
      float const dy = (IW_2 + p(1) - iy) / W_2;
      float const dy2 = dy * dy;
      for (Index ix = 0; ix < (W + 2); ix++) {
        float const dx = (IW_2 + p(0) - ix) / W_2;
        float const dx2 = dx * dx;
        k(ix, iy) = (dx2 + dy2);
      }
    }

    return k;
  }
};

template <size_t W>
struct NewSizedKernel<3, W> : NewKernel
{
  using KTensor = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::declval<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W + 2)...>>>();
  }(std::make_index_sequence<3>()));
  using KPoint = Eigen::Matrix<float, 3, 1>;

  auto distSq(KPoint const p) const -> KTensor
  {
    // Yes, below is a weird mix of integer and floating point division.
    // But it works
    KTensor k;
    constexpr float W_2 = W / 2.f;
    constexpr Index IW_2 = (W + 2) / 2;
    for (Index iz = 0; iz < (W + 2); iz++) {
      float const dz = (IW_2 + p(2) - iz) / W_2;
      float const dz2 = dz * dz;
      for (Index iy = 0; iy < (W + 2); iy++) {
        float const dy = (IW_2 + p(1) - iy) / W_2;
        float const dy2 = dy * dy;
        for (Index ix = 0; ix < (W + 2); ix++) {
          float const dx = (IW_2 + p(0) - ix) / W_2;
          float const dx2 = dx * dx;
          k(ix, iy, iz) = (dx2 + dy2 + dz2);
        }
      }
    }

    return k;
  }
};

template struct NewSizedKernel<1, 3>;
template struct NewSizedKernel<1, 5>;
template struct NewSizedKernel<1, 7>;
template struct NewSizedKernel<2, 3>;
template struct NewSizedKernel<2, 5>;
template struct NewSizedKernel<2, 7>;
template struct NewSizedKernel<3, 3>;
template struct NewSizedKernel<3, 5>;
template struct NewSizedKernel<3, 7>;

std::unique_ptr<NewKernel> make_new_kernel(std::string const &name, bool const is3D, float const osamp)
{
  std::string const type = name.substr(0, 2);
  Index const W = std::stoi(name.substr(2, 1));

  if (is3D) {
    switch (W) {
    case 3:
      if (type == "FI") return std::make_unique<NewFlatIron<3, 3>>(osamp);
    case 5:
      if (type == "FI") return std::make_unique<NewFlatIron<3, 5>>(osamp);
    case 7:
      if (type == "FI") return std::make_unique<NewFlatIron<3, 7>>(osamp);
    }
  } else {
    switch (W) {
    case 3:
      if (type == "FI") return std::make_unique<NewFlatIron<2, 3>>(osamp);
    case 5:
      if (type == "FI") return std::make_unique<NewFlatIron<2, 5>>(osamp);
    case 7:
      if (type == "FI") return std::make_unique<NewFlatIron<2, 7>>(osamp);
    }
  }
  Log::Fail("Unknown kernel type {} or width {}", type, W);
}

} // namespace rl
