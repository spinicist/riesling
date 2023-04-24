#include "scaling.hpp"

#include "algo/otsu.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include <algorithm>
#include <scn/scn.h>

namespace rl {

auto Scaling(args::ValueFlag<std::string> &type, std::shared_ptr<ReconOp> const recon, Cx4 const &data) -> float
{
  float scale;
  if (type.Get() == "bart") {
    Re4 abs = (recon->adjoint(data)).abs();
    auto vec = CollapseToArray(abs);
    std::sort(vec.begin(), vec.end());
    float const med = vec[vec.size() * 0.5];
    float const max = vec[vec.size() - 1];
    float const p90 = vec[vec.size() * 0.9];
    scale = 1.f / (((max - p90) < 2.f * (p90 - med)) ? p90 : max);
    Log::Print("Automatic scaling={}. 50% {} 90% {} 100% {}.", scale, med, p90, max);
  } else if (type.Get() == "otsu") {
    Re4 const abs = (recon->adjoint(data)).abs();
    auto const [thresh, count] = Otsu(CollapseToArray(abs));
    std::vector<float> vals(count);
    std::copy_if(abs.data(), abs.data() + abs.size(), vals.begin(), [thresh=thresh](float const f) { return f > thresh; });
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
