#pragma once

namespace rl {
struct ReconOpts
{
  bool tophat, decant, lowmem;
};

struct f0Opts
{
  float τacq;
  Index Nτ;
};

} // namespace rl
