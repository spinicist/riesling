#include "norms.hpp"

#include "../algo/common.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

auto L1::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1>(λ, sz); }
auto L1::Make(float const λ, CMap b, Ops::Op::Ptr P) -> Prox::Ptr { return std::make_shared<L1>(λ, b, P); }

L1::L1(float const λ_, Index const sz_)
  : Prox(sz_)
  , λ{λ_}
  , b{nullptr, 0}
{
  Log::Print("L1Prox", "λ {}", λ);
}

L1::L1(float const λ_, CMap b_, Ops::Op::Ptr P_)
  : Prox(b_.size())
  , λ{λ_}
  , b{b_}
  , P{P_}
{
  if (P && P->rows() != b.size()) {
    throw(Log::Failure("L1Prox", "Preconditioner size {} did not match data size {}", P->rows(), b.size()));
  }

  Log::Print("L1Prox", "λ {} with bias", λ);
}

void L1::apply(float const α, Map x) const
{
  float const t = α * λ;
  float const nx = ParallelNorm(x);
  Threads::ChunkFor(
    [t, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const ax = std::abs(x[ii]);
        x[ii] = ax > t ? (1.f - t / ax) * x[ii] : 0.f;
      }
    },
    x.size());
  Log::Debug("L1Prox", "α {:4.3E} λ {:4.3E} t {:4.3E} |x| {:4.3E} |z| {:4.3E}", α, λ, t, nx, ParallelNorm(x));
}

void L1::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  if (b.size()) {
    if (P) {
      P->iforward(b, x, -(α * λ));
    } else {
      x.device(Threads::CoreDevice()) = x - (α * λ) * b;
    }
  }
  x.device(Threads::CoreDevice()) = λ * x.array() / x.array().abs().max(λ).cast<Cx>();
  Log::Debug("L1Prox", "Conjugate λ {:4.3E} |x| {:4.3E} |z| {:4.3E}", λ, nx, ParallelNorm(x));
}

auto L1I::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1I>(λ, sz); }

L1I::L1I(float const λ_, Index const sz_)
  : Prox(sz_)
  , λ{λ_}
{
  Log::Print("L1Prox", "λ {}", λ);
}

void L1I::apply(float const α, Map x) const
{
  float const t = α * λ;
  float const nx = ParallelNorm(x);
  Threads::ChunkFor(
    [t, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const xr = x[ii].real();
        float const xi = x[ii].imag();
        float const ax = std::abs(xi);
        x[ii] = Cx(xr, ax > t ? (1.f - t / ax) * xi : 0.f);
      }
    },
    x.size());
  Log::Debug("L1IProx", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(x));
}

void L1I::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x); // Cursed users might do this in place and overwrite x
  Threads::ChunkFor(
    [λ = this->λ, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const xi = x[ii].imag();
        x[ii] = Cx(0.f, λ * xi / std::max(std::abs(xi), λ));
      }
    },
    x.size());
  Log::Debug("L1IProx", "Conjugate λ {} |x| {} |z| {}", λ, nx, ParallelNorm(x));
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

template <int O, int D> void L2<O, D>::apply(float const α, Map x) const
{
  Eigen::TensorMap<CxN<O>> xm(x.data(), shape);
  float const              nx = ParallelNorm(x);
  Index const              nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                                          [sh = this->shape](Index const id) { return sh[id]; });
  Index const              nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                                           [sh = this->shape](Index const id) { return sh[id]; });
  Sz<O> const              shuff = Concatenate(normDims, otherDims);
  Sz<D + 1>                rsh;
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
          xblk *= xblk.constant(1.f - t / norm);
        } else {
          xblk.setZero();
        }
      }
    },
    nBlocks);
  Log::Debug("L2Prox", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(x));
}

template <int O, int D> void L2<O, D>::conj(float const, Map x) const
{
  Eigen::TensorMap<CxN<O>> xm(x.data(), shape);
  float const              nx = ParallelNorm(x);

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
        if (norm > λ) { xblk *= xblk.constant(λ / norm); }
      }
    },
    nBlocks);
  Log::Debug("L2Prox", "Conjugate λ {} |x| {} |z| {}", λ, nx, ParallelNorm(x));
}

template struct L2<1, 1>;
template struct L2<5, 1>;
template struct L2<5, 2>;
template struct L2<6, 1>;
template struct L2<6, 2>;
template struct L2<6, 3>;

auto SumOfSquares::Make(CMap b_, Ops::Op::Ptr P_) -> Prox::Ptr { return std::make_shared<SumOfSquares>(b_, P_); }

SumOfSquares::SumOfSquares(CMap b_, Ops::Op::Ptr P_)
  : Prox(b_.size())
  , b{b_}
  , P{P_}
{
  if (P && P->rows() != b.size()) {
    throw(Log::Failure("SoS", "Preconditioner size {} did not match data size {}", P->rows(), b.size()));
  }
  Log::Print("SoS", "|b| {}", ParallelNorm(b));
}

void SumOfSquares::apply(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  /* Worked this out by following §2.2 of Parykh and Boyd */
  x.device(Threads::CoreDevice()) = (x - b) / (1.f + α) + b;
  Log::Debug("SoS", "α {} |x| {} |z| {}", α, nx, ParallelNorm(x));
}

void SumOfSquares::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  if (P) {
    P->iforward(b, x, -α);
    P->inverse(CMap(x.data(), x.size()), x, α, 1.f);
  } else {
    x.device(Threads::CoreDevice()) = (x - α * b) / (1.f + α);
  }
  Log::Debug("SoS", "α {} |x| {} |z| {}", α, nx, ParallelNorm(x));
}

} // namespace rl::Proxs
