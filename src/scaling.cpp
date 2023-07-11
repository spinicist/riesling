#include "scaling.hpp"

#include "algo/lsmr.hpp"
#include "algo/otsu.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include <algorithm>
#include <scn/scn.h>

namespace rl {

auto Scaling(
  args::ValueFlag<std::string> &type, std::shared_ptr<ReconOp> const A, std::shared_ptr<Ops::Op<Cx>> const P, Cx *const b)
  -> float
{
  float scale;
  if (type.Get() == "bart") {
    LSMR           lsmr{A, P, 2};
    Eigen::ArrayXf x = lsmr.run(b).array().abs();
    std::sort(x.begin(), x.end());
    float const med = x[x.size() * 0.5];
    float const max = x[x.size() - 1];
    float const p90 = x[x.size() * 0.9];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print("Automatic scaling={}. 50% {} 90% {} 100% {}.", scale, med, p90, max);
  } else if (type.Get() == "otsu") {
    LSMR                 lsmr{A, P, 2};
    Eigen::ArrayXf const x = lsmr.run(b).array().abs();
    auto const [thresh, count] = Otsu(x);
    std::vector<float> vals(count);
    std::copy_if(x.data(), x.data() + x.size(), vals.begin(), [thresh = thresh](float const f) { return f > thresh; });
    std::sort(vals.begin(), vals.end());
    float const med = vals[count * 0.5];
    scale = 1.f / med;
    Log::Print("Otsu + median scaling = {}", scale);
  } else {
    if (scn::scan(type.Get(), "{}", scale)) {
      Log::Print("Scale: {}", scale);
    } else {
      Log::Fail("Could not read number from scaling: ", type.Get());
    }
  }
  return scale;
}

} // namespace rl
