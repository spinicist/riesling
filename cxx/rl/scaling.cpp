#include "scaling.hpp"

#include "algo/lsmr.hpp"
#include "algo/otsu.hpp"
#include "algo/stats.hpp"
#include "log/log.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <scn/scan.h>

namespace rl {

auto ScaleImages(std::string const &type, Cx5 const &xx) -> float
{
  float scale;
  if (type == "none") {
    scale = 1.f;
  } else if (type == "bart") {
    Eigen::ArrayXf const x = CollapseToConstVector(xx).array().abs();
    auto const           ps = Percentiles(CollapseToArray(x), {0.5, 0.9, 1.0});
    float const          med = ps[0];
    float const          p90 = ps[1];
    float const          max = ps[2];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print("Scale", "Automatic scaling={}. 50% {} 90% {} 100% {}.", scale, med, p90, max);
  } else if (type == "otsu") {
    Eigen::ArrayXf const x = CollapseToConstVector(xx).array().abs();
    auto const           masked = OtsuMask(x);
    float const          med = Percentiles(CollapseToArray(masked), {0.5}).front();
    scale = 1.f / med;
    Log::Print("Scale", "Otsu + median value {:4.3E} scaling = {:4.3E}", med, scale);
  } else {
    if (auto result = scn::scan<float>(type, "{}")) {
      scale = result->value();
      Log::Print("Scale", "Scale: {}", scale);
    } else {
      throw Log::Failure("Scale", "Could not read number from scaling: ", type);
    }
  }
  return scale;
}

auto ScaleData(std::string const &type, Ops::Op::Ptr const A, Ops::Op::Ptr const P, Ops::Op::Map const b) -> float
{
  float scale;
  if (type == "none") {
    scale = 1.f;
  } else if (type == "bart") {
    LSMR           lsmr{A, P, nullptr, {2}};
    Eigen::ArrayXf x = lsmr.run(b).array().abs();
    std::sort(x.begin(), x.end());
    float const med = x[x.size() * 0.5];
    float const max = x[x.size() - 1];
    float const p90 = x[x.size() * 0.9];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print("Scale", "Automatic scaling={}. 50% {} 90% {} 100% {}.", scale, med, p90, max);
  } else if (type == "otsu") {
    LSMR                 lsmr{A, P, nullptr, {2}};
    Eigen::ArrayXf const x = lsmr.run(b).array().abs();
    auto const           masked = OtsuMask(x);
    float const          med = Percentiles(CollapseToArray(masked), {0.5}).front();
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
  return scale;
}

} // namespace rl
