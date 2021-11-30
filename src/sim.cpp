#include "sim.h"

namespace Sim {

float Parameter::value(Index ii) const
{
  if (logspaced) {
    return logspace(ii);
  } else {
    return linspace(ii);
  }
}

float Parameter::linspace(Index ii) const
{
  assert(ii < N);
  if (N == 1) {
    return (lo + hi) / 2.f;
  } else {
    float const frac = ii / (N - 1.f);
    return lo + frac * (hi - lo);
  }
}

float Parameter::logspace(Index ii) const
{
  assert(ii < N);
  if (N == 1) {
    return (lo + hi) / 2.f;
  } else {
    float const ref = (lo == 0.f) ? 0.1 : std::abs(lo);
    float const frac = ii / (N - 1.f);
    return ref * std::pow((hi - lo) / ref, frac) - (ref - lo);
  }
}

} // namespace Sim
