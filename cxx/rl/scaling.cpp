#include "scaling.hpp"

#include "algo/lsmr.hpp"
#include "algo/otsu.hpp"
#include "algo/stats.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <scn/scan.h>

namespace rl {

auto ScaleData(std::string const &type, Ops::Op<Cx>::Ptr const A, Ops::Op<Cx>::Ptr const P, Ops::Op<Cx>::Map b) -> float
{
  float scale;
  if (type == "none") {
    return 1.f;
  } else if (type == "bart") {
    LSMR           lsmr{A, P, nullptr, 2};
    Eigen::ArrayXf x = lsmr.run(b).array().abs();
    std::sort(x.begin(), x.end());
    float const med = x[x.size() * 0.5];
    float const max = x[x.size() - 1];
    float const p90 = x[x.size() * 0.9];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print("Scale", "Automatic scaling={}. 50% {} 90% {} 100% {}.", scale, med, p90, max);
  } else if (type == "otsu") {
    LSMR                 lsmr{A, P, nullptr, 2};
    Eigen::ArrayXf const x = lsmr.run(b).array().abs();
    auto const masked = OtsuMask(x);
    float const med = Percentiles(masked, {0.5}).front();
    scale = 1.f / med;
    Log::Print("Scale", "Otsu + median scaling = {}", scale);
  } else {
    if (auto result = scn::scan<float>(type, "{}")) {
      scale = result->value();
      Log::Print("Scale", "Scale: {}", scale);
    } else {
      throw Log::Failure("Scale", "Could not read number from scaling: ", type);
    }
  }
  b.device(Threads::CoreDevice()) = b * Cx(scale);
  return scale;
}

void UnscaleData(float const scale, Ops::Op<Cx>::Vector &b)
{
  if (scale != 1.f) {
    b.device(Threads::CoreDevice()) = b / Cx(scale);
  }
  return;
}

} // namespace rl
