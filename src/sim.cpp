#include "sim.h"

namespace Sim {

float Parameter::value(long ii) const {
  if (logspaced) {
    return logspace(ii);
  } else {
    return linspace(ii);
  }
}

float Parameter::linspace(long ii) const {
  assert(ii < N);
  if (N == 1) {
    return (lo + hi) / 2.f;
  } else {
    float const frac = ii / (N - 1.f);
    return lo + frac * (hi - lo);
  }
}

float Parameter::logspace(long ii) const {
  assert(ii < N);
  if (N == 1) {
    return (lo + hi) / 2.f;
  } else {
    float const ref = 0.1;
    float const frac = ii / (N - 1.f);
    return ref * std::pow((hi - lo) / ref, frac) - (ref - lo);
  }
}

}
