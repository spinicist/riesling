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

void L1::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float t = λ * realα->scale;
    Threads::ChunkFor(
      [t, &x, &z](Index lo, Index hi) {
        for (Index ii = lo; ii < hi; ii++) {
          float const ax = std::abs(x[ii]);
          z[ii] = ax < t ? 0.f : (1.f - t / ax) * x[ii];
        }
      },
      x.size());
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("Prox", "Soft Threshold λ {} t {} |x| {} |z| {}", λ, t, ParallelNorm(x), ParallelNorm(z));
    }
  } else {
    throw Log::Failure("Prox", "C++ is stupid");
  }
}

L2::L2(float const λ_, Index const sz_, Index const blk)
  : Prox<Cx>(sz_)
  , λ{λ_}
  , blockSize{blk}
{
  if (sz_ % blockSize != 0) { throw Log::Failure("Prox", "Block size {} does not cleanly divide {}", blockSize, sz_); }
  if (blockSize == 0) { blockSize = sz_; }
  Log::Print("Prox", "L2 Prox λ {} scaled λ {} block size {}", λ_, λ, blockSize);
}

void L2::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  auto const  blks = x.rows() / blockSize;
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto const n = PairwiseNorm(x.segment(ib * blockSize, blockSize));
        if (n > t) {
          z.segment(ib * blockSize, blockSize) = x.segment(ib * blockSize, blockSize) * (1.f - t / n);
        } else {
          z.segment(ib * blockSize, blockSize).setZero();
        }
      }
    },
    blks);
  if (Log::CurrentLevel() == Log::Level::Debug) {
    Log::Debug("Prox", "L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(z));
  }
}

void L2::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float      t = λ * realα->scale;
    auto const blks = x.rows() / blockSize;
    Threads::ChunkFor(
      [&](Index lo, Index hi) {
        for (Index ib = lo; ib < hi; ib++) {
          auto const n = PairwiseNorm(x.segment(ib * blockSize, blockSize));
          if (n > t) {
            z.segment(ib * blockSize, blockSize) = x.segment(ib * blockSize, blockSize) * (1.f - t / n);
          } else {
            z.segment(ib * blockSize, blockSize).setZero();
          }
        }
      },
      blks);
    if (Log::CurrentLevel() == Log::Level::Debug) {
      Log::Debug("Prox", "L2 Prox λ {} t {} |x| {} |z| {}", λ, t, ParallelNorm(x), ParallelNorm(z));
    }
  } else {
    throw Log::Failure("Prox", "C++ is stupid");
  }
}

} // namespace rl::Proxs
