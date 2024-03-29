#include "kernel.hpp"

#include "expsemi.hpp"
#include "kaiser.hpp"
#include "nn.hpp"
#include "radial.hpp"
#include "rectilinear.hpp"
#include "triangle.hpp"

namespace rl {

template <typename Scalar, int ND>
auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<Scalar, ND>>
{
  if (kType == "NN") {
    return std::make_shared<NearestNeighbour<Scalar, ND>>();
  } else if (kType.size() == 7 && kType.substr(0, 4) == "rect") {
    std::string const type = kType.substr(4, 2);
    int const      W = std::stoi(kType.substr(6, 1));
    if (type == "ES") {
      switch (W) {
      case 3: return std::make_shared<Rectilinear<Scalar, ND, ExpSemi<3>>>(osamp);
      case 4: return std::make_shared<Rectilinear<Scalar, ND, ExpSemi<4>>>(osamp);
      case 5: return std::make_shared<Rectilinear<Scalar, ND, ExpSemi<5>>>(osamp);
      case 7: return std::make_shared<Rectilinear<Scalar, ND, ExpSemi<7>>>(osamp);
      default: Log::Fail("Unsupported kernel width {}", W);
      }
    } else if (type == "KB") {
      switch (W) {
      case 3: return std::make_shared<Rectilinear<Scalar, ND, KaiserBessel<3>>>(osamp);
      case 4: return std::make_shared<Rectilinear<Scalar, ND, KaiserBessel<4>>>(osamp);
      case 5: return std::make_shared<Rectilinear<Scalar, ND, KaiserBessel<5>>>(osamp);
      case 7: return std::make_shared<Rectilinear<Scalar, ND, KaiserBessel<7>>>(osamp);
      default: Log::Fail("Unsupported kernel width {}", W);
      }
    }
  } else if (kType.size() == 3) {
    std::string const type = kType.substr(0, 2);
    int const      W = std::stoi(kType.substr(2, 1));
    if (type == "ES") {
      switch (W) {
      case 3: return std::make_shared<Radial<Scalar, ND, ExpSemi<3>>>(osamp);
      case 4: return std::make_shared<Radial<Scalar, ND, ExpSemi<4>>>(osamp);
      case 5: return std::make_shared<Radial<Scalar, ND, ExpSemi<5>>>(osamp);
      case 7: return std::make_shared<Radial<Scalar, ND, ExpSemi<7>>>(osamp);
      default: Log::Fail("Unsupported kernel width {}", W);
      }
    } else if (type == "KB") {
      switch (W) {
      case 3: return std::make_shared<Radial<Scalar, ND, KaiserBessel<3>>>(osamp);
      case 4: return std::make_shared<Radial<Scalar, ND, KaiserBessel<4>>>(osamp);
      case 5: return std::make_shared<Radial<Scalar, ND, KaiserBessel<5>>>(osamp);
      case 7: return std::make_shared<Radial<Scalar, ND, KaiserBessel<7>>>(osamp);
      default: Log::Fail("Unsupported kernel width {}", W);
      }
    }
  }

  Log::Fail("Unknown kernel type {}", kType);
}

template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<float, 1>>;
template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<float, 2>>;
template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<float, 3>>;
template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<Cx, 1>>;
template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<Cx, 2>>;
template auto make_kernel(std::string const &kType, float const osamp) -> std::shared_ptr<Kernel<Cx, 3>>;

} // namespace rl