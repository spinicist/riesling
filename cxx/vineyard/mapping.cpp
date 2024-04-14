#include "mapping.hpp"

#include <cfenv>
#include <cmath>
#include <log.hpp>
#include <ranges>
#include <tl/chunk.hpp>
#include <tl/to.hpp>

#include "tensorOps.hpp"

namespace rl {

template <int NDims>
auto Mapping<NDims>::Bucket::empty() const -> bool
{
  return indices.empty();
}

template <int NDims>
auto Mapping<NDims>::Bucket::size() const -> Index
{
  return indices.size();
}

template <int NDims>
auto Mapping<NDims>::Bucket::bucketSize() const -> Sz<NDims>
{
  Sz<NDims> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

template <int NDims>
auto Mapping<NDims>::Bucket::bucketStart() const -> Sz<NDims>
{
  Sz<NDims> st;
  for (int ii = 0; ii < NDims; ii++) {
    if (minCorner[ii] < 0) {
      st[ii] = -minCorner[ii];
    } else {
      st[ii] = 0L;
    }
  }
  return st;
}

template <int NDims>
auto Mapping<NDims>::Bucket::gridStart() const -> Sz<NDims>
{
  Sz<NDims> st;
  for (int ii = 0; ii < NDims; ii++) {
    if (minCorner[ii] < 0) {
      st[ii] = 0L;
    } else {
      st[ii] = minCorner[ii];
    }
  }
  return st;
}

template <int NDims>
auto Mapping<NDims>::Bucket::sliceSize() const -> Sz<NDims>
{
  Sz<NDims> sl;
  for (int ii = 0; ii < NDims; ii++) {
    if (maxCorner[ii] >= gridSize[ii]) {
      sl[ii] = gridSize[ii] - minCorner[ii];
    } else {
      sl[ii] = maxCorner[ii] - minCorner[ii];
    }
    if (minCorner[ii] < 0) { sl[ii] += minCorner[ii]; }
  }
  return sl;
}

// Helper function to convert a floating-point vector-like expression to integer values
template <typename T>
inline decltype(auto) nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return std::nearbyint(e); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
template <int NDims>
Sz<NDims> fft_size(Sz<NDims> const x, float const os)
{
  // return std::ceil(x);
  Sz<NDims> fsz;
  for (int ii = 0; ii < NDims; ii++) {
    auto ox = x[ii] * os;
    if (ox > 8.f) {
      fsz[ii] = (std::lrint(ox) + 7L) & ~7L;
    } else {
      fsz[ii] = (Index)std::ceil(ox);
    }
  }
  return fsz;
}

// Helper function to sort the cartesian indices
template <size_t N>
std::vector<int32_t> sort(std::vector<std::array<int16_t, N>> const &cart)
{
  std::vector<int32_t> sorted(cart.size());
  std::iota(sorted.begin(), sorted.end(), 0);
  std::sort(sorted.begin(), sorted.end(), [&](Index const a, Index const b) {
    auto const &ac = cart[a];
    auto const &bc = cart[b];
    for (int ii = N - 1; ii > -1; ii--) {
      if (ac[ii] < bc[ii]) {
        return true;
      } else if (ac[ii] > bc[ii]) {
        return false;
      }
    }
    return false;
  });
  return sorted;
}

inline auto Wrap(Index const index, Index const sz) -> Index
{
  Index const t = index + sz;
  Index const w = t - sz * (t / sz);
  return w;
}

template <int NDims>
Mapping<NDims>::Mapping(Trajectory const &traj, float const nomOS, Index const kW, Index const bucketSz, Index const splitSize)
{
  nomDims = FirstN<NDims>(traj.matrix());
  cartDims = Mul(FirstN<NDims>(traj.matrix()), nomOS);
  osamp = cartDims[0] / (float)traj.matrix()[0];
  Log::Print("{}D Mapping, {} samples {} traces. Matrix {} Grid {}", traj.nDims(), traj.nSamples(), traj.nTraces(), nomDims,
             cartDims);

  noncartDims = Sz2{traj.nSamples(), traj.nTraces()};

  Sz<NDims> nB;
  for (int ii = 0; ii < NDims; ii++) {
    nB[ii] = std::ceil(cartDims[ii] / float(bucketSz));
  }
  buckets.reserve(Product(nB));

  if constexpr (NDims == 3) {
    for (Index iz = 0; iz < nB[2]; iz++) {
      for (Index iy = 0; iy < nB[1]; iy++) {
        for (Index ix = 0; ix < nB[0]; ix++) {
          buckets.push_back(
            Bucket{.gridSize = cartDims,
                   .minCorner = Sz3{ix * bucketSz - (kW / 2), iy * bucketSz - (kW / 2), iz * bucketSz - (kW / 2)},
                   .maxCorner = Sz3{std::min((ix + 1) * bucketSz, cartDims[0]) + (kW / 2),
                                    std::min((iy + 1) * bucketSz, cartDims[1]) + (kW / 2),
                                    std::min((iz + 1) * bucketSz, cartDims[2]) + (kW / 2)}});
        }
      }
    }
  } else if constexpr (NDims == 2) {
    for (Index iy = 0; iy < nB[1]; iy++) {
      for (Index ix = 0; ix < nB[0]; ix++) {
        buckets.push_back(Bucket{.gridSize = cartDims,
                                 .minCorner = Sz2{ix * bucketSz - (kW / 2), iy * bucketSz - (kW / 2)},
                                 .maxCorner = Sz2{std::min((ix + 1) * bucketSz, cartDims[0]) + (kW / 2),
                                                  std::min((iy + 1) * bucketSz, cartDims[1]) + (kW / 2)}});
      }
    }
  } else {
    for (Index ix = 0; ix < nB[0]; ix++) {
      buckets.push_back(Bucket{.gridSize = cartDims,
                               .minCorner = Sz1{ix * bucketSz - (kW / 2)},
                               .maxCorner = Sz1{std::min((ix + 1) * bucketSz, cartDims[0]) + (kW / 2)}});
    }
  }

  std::fesetround(FE_TONEAREST);
  auto const center = Div(cartDims, 2);
  int32_t    index = 0;
  Index      invalids = 0;
  for (int32_t is = 0; is < traj.nTraces(); is++) {
    for (int16_t ir = 0; ir < traj.nSamples(); ir++) {
      Re1 const p = traj.point(ir, is);
      if (!B0(p.isfinite().all())()) {
        invalids++;
        continue;
      }
      Eigen::Array<float, NDims, 1> xyz;
      for (int ii = 0; ii < NDims; ii++) {
        xyz[ii] = p[ii] * osamp + center[ii];
      }
      auto const gp = nearby(xyz);
      auto const off = xyz - gp.template cast<float>();

      std::array<int16_t, NDims> ijk;
      for (int ii = 0; ii < NDims; ii++) {
        ijk[ii] = Wrap(gp[ii], cartDims[ii]);
      }
      cart.push_back(ijk);
      offset.push_back(off);
      noncart.push_back(NoncartesianIndex{.trace = is, .sample = ir});

      // Calculate bucket
      Index ib = 0;
      for (int ii = NDims - 1; ii >= 0; ii--) {
        ib = ib * nB[ii] + (ijk[ii] / bucketSz);
      }
      buckets[ib].indices.push_back(index);
      index++;
    }
  }
  Log::Print("Ignored {} invalid trajectory points", invalids);

  std::vector<Bucket> chunked;
  for (auto &bucket : buckets) {
    if (bucket.size() > splitSize) {
      for (auto const indexChunk : tl::views::chunk(bucket.indices, splitSize)) {
        chunked.push_back(Bucket{.gridSize = bucket.gridSize,
                                 .minCorner = bucket.minCorner,
                                 .maxCorner = bucket.maxCorner,
                                 .indices = indexChunk | tl::to<std::vector<int32_t>>()});
      }
      bucket.indices.clear();
    }
  }

  Index const eraseCount = std::erase_if(buckets, [](Bucket const &b) { return b.empty(); });
  buckets.insert(buckets.end(), chunked.begin(), chunked.end());
  Log::Print("Added {} extra, removed {} empty buckets, {} remaining", chunked.size(), eraseCount, buckets.size());
  Log::Print("Total points {}", std::accumulate(buckets.begin(), buckets.end(), 0L,
                                                [](Index sum, Bucket const &b) { return b.indices.size() + sum; }));
  sortedIndices = sort(cart);
}

template struct Mapping<1>;
template struct Mapping<2>;
template struct Mapping<3>;

} // namespace rl
