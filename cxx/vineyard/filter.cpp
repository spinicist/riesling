#include "filter.hpp"

#include "log.hpp"
#include "tensors.hpp"
#include "threads.hpp"

#include <functional>

namespace rl {
inline float Tukey(float const &r, float const &sw, float const &ew, float const &eh)
{
  if (r > ew) {
    return 0.f;
  } else if (r > sw) {
    return (0.5f * ((1 + eh) + (1 - eh) * cos((M_PI * (r - sw)) / (ew - sw))));
  } else {
    return 1.f;
  }
}

template <typename Scalar, int D>
void KSFilter(std::function<float(float const &)> const &f, Eigen::Tensor<Scalar, D> &ks)
{
  auto const sz = ks.dimension(D - 1);
  auto const sy = ks.dimension(D - 2);
  auto const sx = ks.dimension(D - 3);

  auto const hz = sz / 2;
  auto const hy = sy / 2;
  auto const hx = sx / 2;

  auto task = [&](Index const iz) {
    for (Index iy = 0; iy < sy; iy++) {
      for (Index ix = 0; ix < sx; ix++) {
        float const rz = static_cast<float>(iz - hz) / hz;
        float const ry = static_cast<float>(iy - hy) / hy;
        float const rx = static_cast<float>(ix - hx) / hx;
        float const r = sqrt(rx * rx + ry * ry + rz * rz);
        float const val = f(r);
        ks.template chip<D - 1>(iz).template chip<D - 2>(iy).template chip<D - 3>(ix) *=
          ks.template chip<D - 1>(iz).template chip<D - 2>(iy).template chip<D - 3>(ix).constant(val);
      }
    }
  };
  Threads::For(task, sz, "Filtering");
}

void CartesianTukey(float const &s, float const &e, float const &h, Cx4 &x)
{
  Log::Print("Applying Tukey filter width {}-{} height {}", s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  KSFilter(f, x);
}

void NoncartesianTukey(float const &s, float const &e, float const &h, Re3 const &coords, Cx4 &x)
{
  auto const nC = x.dimension(0);
  auto const nS = x.dimension(1);
  auto const nT = x.dimension(2);
  auto const nSlice = x.dimension(3);
  assert(coords.dimension(1) == nS);
  assert(coords.dimension(2) == nT);
  Log::Print("Noncartesian Tukey start {} end {} height {}", s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  x.device(Threads::GlobalDevice()) =
    x * coords.square().sum(Sz1{0}).sqrt().unaryExpr(f).reshape(Sz4{1, nS, nT, 1}).broadcast(Sz4{nC, 1, 1, nSlice});
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
  Log::Print("Noncartesian Tukey start {} end {} height {}", s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  x.device(Threads::GlobalDevice()) =
    x * coords.square().sum(Sz1{0}).sqrt().unaryExpr(f).reshape(Sz5{1, nS, nT, 1, 1}).broadcast(Sz5{nC, 1, 1, nSlice, nV});
}

} // namespace rl
