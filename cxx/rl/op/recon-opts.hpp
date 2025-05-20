#pragma once

namespace rl {
struct ReconOpts
{
  bool decant, lowmem;
};

struct f0Opts
{
  float τacq;
  Index Nτ;
};

} // namespace rl
