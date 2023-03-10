#include "scaling.hpp"

#include "log.hpp"
#include <scn/scn.h>

namespace rl {

auto Scaling(args::ValueFlag<std::string> &scale, std::shared_ptr<ReconOp> const &recon, Cx5 const &data) -> float
{
  float val;
  if (scale.Get() == "auto") {
    Log::Print(FMT_STRING("Automatic scaling."));
    Cx4 const temp = recon->cadjoint(CChipMap(data, 0));
    val = std::sqrt(temp.size()) / Norm(temp);
  } else {
    if (!scn::scan(scale.Get(), "{}", val)) {
      Log::Fail(FMT_STRING("Could not read number from scaling: "), scale.Get());
    }
  }
  Log::Print(FMT_STRING("Scale: {}"), val);
  return val;
}

} // namespace rl
