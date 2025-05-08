#include "filter.hpp"

#include "log/log.hpp"
#include "sys/threads.hpp"
#include "tensors.hpp"

#include <cmath>
#include <functional>
#include <numbers>

namespace rl {
inline float Tukey(float const &r, float const &sw, float const &ew, float const &eh)
{
  if (r > ew) {
    return 0.f;
  } else if (r > sw) {
    return (0.5f * ((1.f + eh) + (1.f - eh) * std::cos((std::numbers::pi_v<float> * (r - sw)) / (ew - sw))));
  } else {
    return 1.f;
  }
}

void NoncartesianTukey(float const &s, float const &e, float const &h, Re3 const &coords, Cx4 &x)
{
  auto const nC = x.dimension(0);
  auto const nS = x.dimension(1);
  auto const nT = x.dimension(2);
  auto const nSlice = x.dimension(3);
  assert(coords.dimension(1) == nS);
  assert(coords.dimension(2) == nT);
  Log::Print("Traj", "NC Tukey width {}-{} height {}", s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  Re2 const   r = coords.square().sum(Sz1{0}).sqrt();
  x.device(Threads::TensorDevice()) =
    r.isfinite().select(x * r.unaryExpr(f).reshape(Sz4{1, nS, nT, 1}).broadcast(Sz4{nC, 1, 1, nSlice}), x.constant(0.f));
}

void NoncartesianTukey(float const &s, float const &e, float const &h, Re3 const &coords, Cx5 &x)
{
  auto const nC = x.dimension(0);
  auto const nS = x.dimension(1);
  auto const nT = x.dimension(2);
  auto const nSlice = x.dimension(3);
  auto const nV = x.dimension(4);
  assert(coords.dimension(1) == nS);
  assert(coords.dimension(2) == nT);
  Log::Print("Traj", "NC Tukey width {}-{} height {}", s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  x.device(Threads::TensorDevice()) =
    x * coords.square().sum(Sz1{0}).sqrt().unaryExpr(f).reshape(Sz5{1, nS, nT, 1, 1}).broadcast(Sz5{nC, 1, 1, nSlice, nV});
}

} // namespace rl
