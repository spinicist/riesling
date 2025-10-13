#pragma once

namespace rl {
struct ReconOpts
{
  bool decant, lowmem;
};

struct f0Opts
{
  float τ0, τacq;
  Index Nτ;
};

} // namespace rl
