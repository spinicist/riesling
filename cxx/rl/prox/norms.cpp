#include "norms.hpp"

#include "../algo/common.hpp"
#include "../log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

L1::L1(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("Prox", "L1 / Soft Threshold λ {}", λ);
}

void L1::apply(float const α, CMap const &x, Map &z) const
{
  float t = α * λ;
  Threads::ChunkFor(
    [t, &x, &z](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const ax = std::abs(x[ii]);
        z[ii] = ax < t ? 0.f : (1.f - t / ax) * x[ii];
      }
    },
    x.size());
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("Prox", "Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

// void L1::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
// {
//   if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
//     float t = λ * realα->scale;
//     Threads::ChunkFor(
//       [t, &x, &z](Index lo, Index hi) {
//         for (Index ii = lo; ii < hi; ii++) {
//           float const ax = std::abs(x[ii]);
//           z[ii] = ax < t ? 0.f : (1.f - t / ax) * x[ii];
//         }
//       },
//       x.size());
//     if (Log::CurrentLevel() == Log::Level::Debug) {
//       Log::Debug("Prox", "Soft Threshold λ {} t {} |x| {} |z| {}", λ, t, ParallelNorm(x), ParallelNorm(z));
//     }
//   } else {
//     throw Log::Failure("Prox", "C++ is stupid");
//   }
// }

template <int O, int D>
L2<O, D>::L2(float const λ_, Sz<O> const &s, Sz<D> const &d)
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

template <int O, int D> void L2<O, D>::apply(float const α, CMap const &x, Map &z) const
{
  Eigen::TensorMap<CxN<O> const> const xm(x.data(), shape);
  Eigen::TensorMap<CxN<O>>             zm(z.data(), shape);

  Index const     nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                                 [shape = this->shape](Index const id) { return shape[id]; });
  Index const     nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                                  [shape = this->shape](Index const id) { return shape[id]; });
  Sz<O> const     shuff = Concatenate(normDims, otherDims);
  Sz<D + 1> rsh;
  std::transform(normDims.cbegin(), normDims.cend(), rsh.begin(), [shape = this->shape](Index const id) { return shape[id]; });
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
          zblk = xblk * xblk.constant(1.f - t / n);
        } else {
          zblk.setZero();
        }
      }
    },
    nBlocks);
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("Prox", "L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

// void L2::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
// {
//   if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
//     float      t = λ * realα->scale;
//     auto const blks = x.rows() / blockSize;
//     Threads::ChunkFor(
//       [&](Index lo, Index hi) {
//         for (Index ib = lo; ib < hi; ib++) {
//           auto const n = PairwiseNorm(x.segment(ib * blockSize, blockSize));
//           if (n > t) {
//             z.segment(ib * blockSize, blockSize) = x.segment(ib * blockSize, blockSize) * (1.f - t / n);
//           } else {
//             z.segment(ib * blockSize, blockSize).setZero();
//           }
//         }
//       },
//       blks);
//     if (Log::CurrentLevel() == Log::Level::Debug) {
//       Log::Debug("Prox", "L2 Prox λ {} t {} |x| {} |z| {}", λ, t, ParallelNorm(x), ParallelNorm(z));
//     }
//   } else {
//     throw Log::Failure("Prox", "C++ is stupid");
//   }
// }

template struct L2<6, 2>;

} // namespace rl::Proxs
