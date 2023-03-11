#include "scaling.hpp"

#include "log.hpp"
#include "tensorOps.hpp"
#include <scn/scn.h>
#include <algorithm>

namespace rl {

auto Scaling(args::ValueFlag<std::string> &scale, std::shared_ptr<ReconOp> const &recon, Cx5 const &data) -> float
{
  float val;
  if (scale.Get() == "auto") {
    Re4 abs = (recon->cadjoint(CChipMap(data, 0))).abs();
    auto vec = CollapseToVector(abs);
    std::sort(vec.begin(), vec.end());
    val = 1.f / vec[vec.size() * 0.9];
    Log::Print(FMT_STRING("Automatic scaling {}"), val);
  } else {
    if (scn::scan(scale.Get(), "{}", val)) {
        Log::Print(FMT_STRING("Scale: {}"), val);
    } else {
      Log::Fail(FMT_STRING("Could not read number from scaling: "), scale.Get());
    }
  }
  return val;
}

} // namespace rl
