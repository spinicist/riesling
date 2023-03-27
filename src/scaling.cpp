#include "scaling.hpp"

#include "log.hpp"
#include "tensorOps.hpp"
#include <algorithm>
#include <scn/scn.h>

namespace rl {

auto Scaling(args::ValueFlag<std::string> &type, std::shared_ptr<ReconOp> const recon, Cx4 const &data) -> float
{
  float scale;
  if (type.Get() == "auto") {
    Re4 abs = (recon->cadjoint(data)).abs();
    auto vec = CollapseToVector(abs);
    std::sort(vec.begin(), vec.end());
    float const med = vec[vec.size() * 0.5];
    float const max = vec[vec.size() - 1];
    float const p90 = vec[vec.size() * 0.9];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print(FMT_STRING("Automatic scaling={}. 50%/90%/100% {}/{}/{}."), scale, med, p90, max);
  } else {
    if (scn::scan(type.Get(), "{}", scale)) {
      Log::Print(FMT_STRING("Scale: {}"), scale);
    } else {
      Log::Fail(FMT_STRING("Could not read number from scaling: "), type.Get());
    }
  }
  return scale;
}

} // namespace rl
