#include "norms.hpp"

#include "../algo/common.hpp"
#include "../log/debug.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

auto L1::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1>(λ, sz); }

L1::L1(float const λ_, Index const sz_)
  : Prox(sz_)
  , λ{λ_}
{
  Log::Print("L1Prox", "λ {}", λ);
}

void L1::apply(float const α, CMap x, Map z) const
{
  float const t = α * λ;
  float const nx = Log::IsDebugging() ? ParallelNorm(x) : 0.f; // Cursed users might do this in place and overwrite x
  Threads::ChunkFor(
    [t, &x, &z](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const ax = std::abs(x[ii]);
        z[ii] = ax > t ? (1.f - t / ax) * x[ii] : 0.f;
      }
    },
    x.size());
  Log::Debug("L1Prox", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(z));
}

void L1::conj(float const, CMap x, Map z) const
{
  /* Amazingly, this doesn't depend on α. Maths is based. */
  Threads::ChunkFor(
    [λ = this->λ, &x, &z](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const a = std::abs(x[ii]);
        z[ii] = a > λ ? x[ii] * λ / a : x[ii];
      }
    },
    x.size());
  Log::Debug("L1Prox", "Conjugate λ {} |x| {} |z| {}", λ, ParallelNorm(x), ParallelNorm(z));
}

template <int O, int D> auto L2<O, D>::Make(float const λ, Sz<O> const &s, Sz<D> const &d) -> Prox::Ptr
{
  return std::make_shared<L2>(λ, s, d);
}

template <int O, int D> L2<O, D>::L2(float const λ_, Sz<O> const &s, Sz<D> const &d)
  : Prox(Product(s))
  , λ{λ_}
  , shape{s}
  , normDims(d)
{
  Log::Print("L2Prox", "λ {} shape {} norm dims {}", λ_, shape, normDims);
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
  float const t = α * λ; // * std::sqrt(nElems);
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > t) {
          zm.shuffle(shuff).reshape(rsh).template chip<D>(ib) = xblk * xblk.constant(1.f - t / norm);
        } else {
          zm.shuffle(shuff).reshape(rsh).template chip<D>(ib).setZero();
        }
      }
    },
    nBlocks);
  if (Log::IsDebugging()) { Log::Print("L2Prox", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z)); }
}

template <int O, int D> void L2<O, D>::conj(float const, CMap x, Map z) const
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

  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > λ) {
          zm.shuffle(shuff).reshape(rsh).template chip<D>(ib) = xblk * xblk.constant(λ / norm);
        } else {
          zm.shuffle(shuff).reshape(rsh).template chip<D>(ib) = xblk;
        }
      }
    },
    nBlocks);
  if (Log::IsDebugging()) { Log::Print("L2Prox", "λ {} |x| {} |z| {}", λ, ParallelNorm(x), ParallelNorm(z)); }
}

template struct L2<1, 1>;
template struct L2<5, 1>;
template struct L2<5, 2>;
template struct L2<6, 1>;
template struct L2<6, 2>;

auto L2Residual::Make(CMap b) -> Prox::Ptr { return std::make_shared<L2Residual>(b); }

L2Residual::L2Residual(CMap b_)
  : Prox(b_.size())
  , b{b_}
{
  Log::Print("L2Res", "|b| {}", ParallelNorm(b));
}

void L2Residual::apply(float const α, CMap x, Map z) const
{
  float const nx = ParallelNorm(x); // Cursed users might do this in place and overwrite x
  /* Worked this out by following §2.2 of Parykh and Boyd */
  z.device(Threads::CoreDevice()) = (x - b) / (1.f + α) + b;
  Log::Debug("L2Res", "α {} |x| {} |z| {}", α, nx, ParallelNorm(z));
}

void L2Residual::conj(float const α, CMap x, Map z) const
{
  float const nx = ParallelNorm(x); // Cursed users might do this in place and overwrite x
  z.device(Threads::CoreDevice()) = (x - α * b) / (1.f + α);
  Log::Debug("L2Res", "α {} |x| {} |z| {}", α, nx, ParallelNorm(z));
}

} // namespace rl::Proxs
