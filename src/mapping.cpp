#include "mapping.hpp"

#include <cfenv>
#include <cmath>
#include <range/v3/range.hpp>
#include <range/v3/view.hpp>

#include "tensorOps.hpp"

namespace rl {

template <size_t Rank>
auto Mapping<Rank>::Bucket::empty() const -> bool
{
  return indices.empty();
}

template <size_t Rank>
auto Mapping<Rank>::Bucket::size() const -> Index
{
  return indices.size();
}

template <size_t Rank>
auto Mapping<Rank>::Bucket::gridSize() const -> Sz<Rank>
{
  Sz<Rank> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
  // return Sz3{maxCorner[0] - minCorner[0], maxCorner[1] - minCorner[1], maxCorner[2] - minCorner[2]};
}

// Helper function to convert a floating-point vector-like expression to integer values
template <typename T>
inline decltype(auto) nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return std::nearbyint(e); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
inline Index fft_size(float const x)
{
  return std::ceil(x);
  // if (x > 8.f) {
  //   return (std::lrint(x) + 7L) & ~7L;
  // } else {
  //   return (Index)std::ceil(x);
  // }
}

// Helper function to sort the cartesian indices
template <size_t N>
std::vector<int32_t> sort(std::vector<std::array<int16_t, N>> const &cart)
{
  auto const start = Log::Now();
  std::vector<int32_t> sorted(cart.size());
  std::iota(sorted.begin(), sorted.end(), 0);
  std::sort(sorted.begin(), sorted.end(), [&](Index const a, Index const b) {
    auto const &ac = cart[a];
    auto const &bc = cart[b];
    for (int ii = N - 1; ii >= 0; ii--) {
      if (ac[ii] < bc[ii]) {
        return true;
      } else if (ac[ii] > bc[ii]) {
        return false;
      }
    }
    return false;
  });
  Log::Print<Log::Level::High>(FMT_STRING("Grid co-ord sorting: {}"), Log::ToNow(start));
  return sorted;
}

template <size_t Rank>
Mapping<Rank>::Mapping(
  Trajectory const &traj,
  Index const kW,
  float const os,
  Index const bucketSz,
  Index const splitSize,
  Index const read0)
{
  Info const &info = traj.info();
  Index const gridSz = fft_size(info.matrix[0] * os);
  Log::Print(FMT_STRING("Mapping to grid size {}"), gridSz);

  std::fill(cartDims.begin(), cartDims.end(), gridSz);
  noncartDims = Sz2{traj.nSamples(), traj.nTraces()};
  frames = traj.nFrames();
  frameWeights = Eigen::ArrayXf(frames);
  frameWeights.setZero();

  Index const nB = std::ceil(gridSz / float(bucketSz));
  buckets.reserve(pow(nB, Rank));

  if constexpr (Rank == 3) {
    for (Index iz = 0; iz < nB; iz++) {
      for (Index iy = 0; iy < nB; iy++) {
        for (Index ix = 0; ix < nB; ix++) {
          buckets.push_back(Bucket{
            Sz3{ix * bucketSz - (kW / 2), iy * bucketSz - (kW / 2), iz * bucketSz - (kW / 2)},
            Sz3{
              std::min((ix + 1) * bucketSz, cartDims[0]) + (kW / 2),
              std::min((iy + 1) * bucketSz, cartDims[1]) + (kW / 2),
              std::min((iz + 1) * bucketSz, cartDims[2]) + (kW / 2)}});
        }
      }
    }
  } else {
    for (Index iy = 0; iy < nB; iy++) {
      for (Index ix = 0; ix < nB; ix++) {
        buckets.push_back(Bucket{
          Sz2{ix * bucketSz - (kW / 2), iy * bucketSz - (kW / 2)},
          Sz2{
            std::min((ix + 1) * bucketSz, cartDims[0]) + (kW / 2),
            std::min((iy + 1) * bucketSz, cartDims[1]) + (kW / 2)}});
      }
    }
  }

  Log::Print("Calculating mapping");
  std::fesetround(FE_TONEAREST);
  float const maxRad = (gridSz / 2) - 1.f;
  Sz<Rank> center;
  std::transform(cartDims.begin(), cartDims.end(), center.begin(), [](Index const c) { return c / 2; });
  int32_t index = 0;
  Index NaNs = 0;
  for (int32_t is = 0; is < traj.nTraces(); is++) {
    auto const fr = traj.frame(is);
    if ((fr >= 0) && (fr < frames)) {
      for (int16_t ir = read0; ir < traj.nSamples(); ir++) {
        Re1 const p = traj.point(ir, is);
        Eigen::Array<float, Rank, 1> xyz;
        for (Index ii = 0; ii < Rank; ii++) {
          xyz[ii] = p[ii] * maxRad * 2.f;
        }
        if (xyz.array().isFinite().all()) { // Allow for NaNs in trajectory for blanking
          auto const gp = nearby(xyz);
          auto const off = xyz - gp.template cast<float>();
          std::array<int16_t, Rank> ijk;
          std::transform(center.begin(), center.end(), gp.begin(), ijk.begin(), [](float const f1, float const f2) {
            return f1 + f2;
          });
          cart.push_back(ijk);
          offset.push_back(off);
          noncart.push_back(NoncartesianIndex{.trace = is, .sample = ir});
          frame.push_back(fr);

          // Calculate bucket
          Index ib = 0;
          for (int ii = Rank - 1; ii >= 0; ii--) {
            ib = ib * nB + (ijk[ii] / bucketSz);
          }
          buckets[ib].indices.push_back(index);
          frameWeights[fr] += 1;
          index++;
        } else {
          NaNs++;
        }
      }
    }
  }
  Log::Print("Ignored {} non-finite trajectory points", NaNs);

  std::vector<Bucket> chunked;
  for (auto &bucket : buckets) {
    if (bucket.size() > splitSize) {
      for (auto const indexChunk : ranges::views::chunk(bucket.indices, splitSize)) {
        chunked.push_back(Bucket{
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

  Log::Print("Total points {}", std::accumulate(buckets.begin(), buckets.end(), 0L, [](Index sum, Bucket const &b) {
               return b.indices.size() + sum;
             }));
  sortedIndices = sort(cart);

  frameWeights = frameWeights.maxCoeff() / frameWeights;
  Log::Print(FMT_STRING("Frame weights: {}"), frameWeights.transpose());
}

template struct Mapping<2>;
template struct Mapping<3>;

} // namespace rl
