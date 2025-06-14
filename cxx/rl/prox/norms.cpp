#include "norms.hpp"

#include "../algo/common.hpp"
#include "../log/debug.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

auto L1::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1>(λ, sz); }

L1::L1(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("Prox", "L1 / Soft Threshold λ {}", λ);
}

void L1::apply(float const α, CMap x, Map z) const
{
  float t = α * λ;
  Threads::ChunkFor(
    [t, &x, &z](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const ax = std::abs(x[ii]);
        z[ii] = ax > t ? (1.f - t / ax) * x[ii] : 0.f;
      }
    },
    x.size());
  if (Log::IsDebugging()) {
    Log::Print("Prox", "Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

template <int O, int D> auto L2<O, D>::Make(float const λ, Sz<O> const &s, Sz<D> const &d) -> Prox::Ptr
{
  return std::make_shared<L2>(λ, s, d);
}

template <int O, int D> L2<O, D>::L2(float const λ_, Sz<O> const &s, Sz<D> const &d)
  : Prox<Cx>(Product(s))
  , λ{λ_}
  , shape{s}
  , normDims(d)
{
  Log::Print("Prox", "L2 Prox λ {} scaled λ {} shape {} norm dims", λ_, λ, shape, normDims);
  Sz<O> all;
  std::iota(all.begin(), all.end(), 0);
  std::set_difference(all.cbegin(), all.cend(), normDims.cbegin(), normDims.cend(), otherDims.begin());
}

template <int O, int D> void L2<O, D>::apply(float const α, CMap x, Map z) const
{
  Eigen::TensorMap<CxN<O> const> const xm(x.data(), shape);
  Eigen::TensorMap<CxN<O>>             zm(z.data(), shape);

  Index const nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                             [sh = this->shape](Index const id) { return sh[id]; });
  Index const nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                              [sh = this->shape](Index const id) { return sh[id]; });
  Sz<O> const shuff = Concatenate(normDims, otherDims);
  Sz<D + 1>   rsh;
  std::transform(normDims.cbegin(), normDims.cend(), rsh.begin(), [sh = this->shape](Index const id) { return sh[id]; });
  rsh[D] = nBlocks;
  Sz<1> const rel{nElems};

  float const t = α * λ * std::sqrt(nElems);
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto       zblk = zm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > t) {
          zblk = xblk * xblk.constant(1.f - t / norm);
        } else {
          zblk.setZero();
        }
      }
    },
    nBlocks);
  if (Log::IsDebugging()) {
    Log::Print("Prox", "L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

template struct L2<6, 1>;
template struct L2<6, 2>;

auto L1Ball::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1Ball>(λ, sz); }

L1Ball::L1Ball(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("Prox", "L1 Ball λ {}", λ);
}

void L1Ball::apply(float const α, CMap x, Map z) const
{
  float t = α * λ;
  Threads::ChunkFor(
    [t, &x, &z](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const a = std::abs(x[ii]);
        z[ii] = a > t ? x[ii] * t / a : x[ii];
      }
    },
    x.size());
  if (Log::IsDebugging()) {
    Log::Print("Prox", "L1 Ball α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

template <int O, int D> auto L2Ball<O, D>::Make(float const λ, Sz<O> const &s, Sz<D> const &d) -> Prox::Ptr
{
  return std::make_shared<L2Ball>(λ, s, d);
}

template <int O, int D> L2Ball<O, D>::L2Ball(float const λ_, Sz<O> const &s, Sz<D> const &d)
  : Prox<Cx>(Product(s))
  , λ{λ_}
  , shape{s}
  , normDims(d)
{
  Log::Print("Prox", "L2 Ball λ {} scaled λ {} shape {} norm dims", λ_, λ, shape, normDims);
  Sz<O> all;
  std::iota(all.begin(), all.end(), 0);
  std::set_difference(all.cbegin(), all.cend(), normDims.cbegin(), normDims.cend(), otherDims.begin());
}

template <int O, int D> void L2Ball<O, D>::apply(float const α, CMap x, Map z) const
{
  Eigen::TensorMap<CxN<O> const> const xm(x.data(), shape);
  Eigen::TensorMap<CxN<O>>             zm(z.data(), shape);

  Index const nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                             [sh = this->shape](Index const id) { return sh[id]; });
  Index const nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                              [sh = this->shape](Index const id) { return sh[id]; });
  Sz<O> const shuff = Concatenate(normDims, otherDims);
  Sz<D + 1>   rsh;
  std::transform(normDims.cbegin(), normDims.cend(), rsh.begin(), [sh = this->shape](Index const id) { return sh[id]; });
  rsh[D] = nBlocks;
  Sz<1> const rel{nElems};

  float const t = α * λ * std::sqrt(nElems);
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto       zblk = zm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > t) {
          zblk = xblk * xblk.constant(t / norm);
        } else {
          zblk = xblk;
        }
      }
    },
    nBlocks);
  if (Log::IsDebugging()) {
    Log::Print("Prox", "L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

template struct L2Ball<6, 1>;
template struct L2Ball<6, 2>;

} // namespace rl::Proxs
