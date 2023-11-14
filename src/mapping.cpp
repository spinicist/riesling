#include "mapping.hpp"

#include <cfenv>
#include <cmath>
#include <range/v3/range.hpp>
#include <range/v3/view.hpp>

#include "tensorOps.hpp"

namespace rl {

template <int Rank>
auto Mapping<Rank>::Bucket::empty() const -> bool
{
  return indices.empty();
}

template <int Rank>
auto Mapping<Rank>::Bucket::size() const -> Index
{
  return indices.size();
}

template <int Rank>
auto Mapping<Rank>::Bucket::bucketSize() const -> Sz<Rank>
{
  Sz<Rank> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

template <int Rank>
auto Mapping<Rank>::Bucket::bucketStart() const -> Sz<Rank>
{
  Sz<Rank> st;
  for (int ii = 0; ii < Rank; ii++) {
    if (minCorner[ii] < 0) {
      st[ii] = -minCorner[ii];
    } else {
      st[ii] = 0L;
    }
  }
  return st;
}

template <int Rank>
auto Mapping<Rank>::Bucket::gridStart() const -> Sz<Rank>
{
  Sz<Rank> st;
  for (int ii = 0; ii < Rank; ii++) {
    if (minCorner[ii] < 0) {
      st[ii] = 0L;
    } else {
      st[ii] = minCorner[ii];
    }
  }
  return st;
}

template <int Rank>
auto Mapping<Rank>::Bucket::sliceSize() const -> Sz<Rank>
{
  Sz<Rank> sl;
  for (int ii = 0; ii < Rank; ii++) {
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
template <int Rank>
Sz<Rank> fft_size(Sz<Rank> const x, float const os)
{
  // return std::ceil(x);
  Sz<Rank> fsz;
  for (int ii = 0; ii < Rank; ii++) {
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
  auto const           start = Log::Now();
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
  Log::Print<Log::Level::High>("Grid co-ord sorting: {}", Log::ToNow(start));
  return sorted;
}

inline auto Wrap(Index const index, Index const sz) -> Index
{
  Index const t = index + sz;
  Index const w = t - sz * (t / sz);
  return w;
}

template <int Rank>
Mapping<Rank>::Mapping(Trajectory const &traj, float const nomOS, Index const kW, Index const bucketSz, Index const splitSize)
{
  Info const &info = traj.info();
  nomDims = FirstN<Rank>(info.matrix);
  cartDims = fft_size<Rank>(FirstN<Rank>(info.matrix), nomOS);
  osamp = cartDims[0] / (float)info.matrix[0];
  Log::Print("{}D Mapping, {} samples {} traces. Matrix {} Grid {}", traj.nDims(), traj.nSamples(), traj.nTraces(), nomDims,
             cartDims);

  noncartDims = Sz2{traj.nSamples(), traj.nTraces()};

  Sz<Rank> nB;
  for (int ii = 0; ii < Rank; ii++) {
    nB[ii] = std::ceil(cartDims[ii] / float(bucketSz));
  }
  buckets.reserve(Product(nB));

  if constexpr (Rank == 3) {
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
  } else if constexpr (Rank == 2) {
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
  Eigen::Array<float, Rank, 1> center, scales;
  for (int ii = 0; ii < Rank; ii++) {
    scales[ii] = cartDims[ii];
    center[ii] = cartDims[ii] / 2;
  }
  int32_t index = 0;
  Index   invalids = 0;
  for (int32_t is = 0; is < traj.nTraces(); is++) {
    for (int16_t ir = 0; ir < traj.nSamples(); ir++) {
      Re1 const p = traj.point(ir, is);
      if (!B0(p.isfinite().all())()) {
        invalids++;
        continue;
      }
      Eigen::Array<float, Rank, 1> xyz;
      for (int ii = 0; ii < Rank; ii++) {
        xyz[ii] = p[ii] * scales[ii] + center[ii];
      }
      auto const gp = nearby(xyz);
      auto const off = xyz - gp.template cast<float>();

      std::array<int16_t, Rank> ijk;
      for (int ii = 0; ii < Rank; ii++) {
        ijk[ii] = Wrap(gp[ii], cartDims[ii]);
      }
      cart.push_back(ijk);
      offset.push_back(off);
      noncart.push_back(NoncartesianIndex{.trace = is, .sample = ir});

      // Calculate bucket
      Index ib = 0;
      for (int ii = Rank - 1; ii >= 0; ii--) {
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
      for (auto const indexChunk : ranges::views::chunk(bucket.indices, splitSize)) {
        chunked.push_back(Bucket{.gridSize = bucket.gridSize,
                                 .minCorner = bucket.minCorner,
                                 .maxCorner = bucket.maxCorner,
                                 .indices = indexChunk | ranges::to<std::vector<int32_t>>()});
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
