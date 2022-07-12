#include "kernel.h"

#include "kernel.hpp"

#include "log.h"

namespace rl {

std::unique_ptr<Kernel> make_kernel(std::string const &k, Info::Type const t, float const os)
{
  if (k == "NN") {
    return std::make_unique<NearestNeighbour>();
  } else if (k == "KB3") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<KaiserBessel<3, 3>>(os);
    } else {
      return std::make_unique<KaiserBessel<3, 1>>(os);
    }
  } else if (k == "KB5") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<KaiserBessel<5, 5>>(os);
    } else {
      return std::make_unique<KaiserBessel<5, 1>>(os);
    }
  } else if (k == "FI3") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<FlatIron<3, 3>>(os);
    } else {
      return std::make_unique<FlatIron<3, 1>>(os);
    }
  } else if (k == "FI5") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<FlatIron<5, 5>>(os);
    } else {
      return std::make_unique<FlatIron<5, 1>>(os);
    }
  } else {
    Log::Fail(FMT_STRING("Unknown kernel type: {}"), k);
  }
}

} // namespace rl
