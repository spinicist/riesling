#include "grid-mapping.hpp"

#include <cfenv>
#include <cmath>
#include <log.hpp>
#include <ranges>
#include <tl/chunk.hpp>
#include <tl/to.hpp>

#include "tensors.hpp"

namespace rl {

// Helper function to convert a floating-point vector-like expression to integer values
template <typename T> inline decltype(auto) nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return (Index)std::nearbyint(e); });
}

// Helper function to sort the cartesian indices
template <size_t N> std::vector<int32_t> sort(std::vector<std::array<int16_t, N>> const &cart)
{
  std::vector<int32_t> sorted(cart.size());
  std::iota(sorted.begin(), sorted.end(), 0);
  std::sort(sorted.begin(), sorted.end(), [&](size_t const a, size_t const b) {
    auto const &ac = cart[a];
    auto const &bc = cart[b];
    for (size_t fi = 0; fi < N; fi++) {
      size_t ii = N - 1 - fi;
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

template <int NDims, bool VCC>
Mapping<NDims, VCC>::Mapping(
  TrajectoryN<NDims> const &traj, float const nomOS, Index const kW, Index const subgridSz, Index const splitSize)
{
  nomDims = traj.matrix();
  cartDims = Mul(nomDims, nomOS);
  osamp = cartDims[0] / (float)traj.matrix()[0];
  Log::Print("Mapping samples {} traces {} OS {} Matrix {} Grid {}", traj.nSamples(), traj.nTraces(), nomOS, nomDims, cartDims);

  noncartDims = Sz2{traj.nSamples(), traj.nTraces()};

  Sz<NDims> nB;
  for (size_t ii = 0; ii < NDims; ii++) {
    nB[ii] = (Index)std::ceil(cartDims[ii] / float(subgridSz));
  }
  subgrids.reserve(static_cast<size_t>(Product(nB)));

  if constexpr (NDims == 3) {
    for (Index iz = 0; iz < nB[2]; iz++) {
      for (Index iy = 0; iy < nB[1]; iy++) {
        for (Index ix = 0; ix < nB[0]; ix++) {
          subgrids.push_back(Subgrid<NDims, VCC>{
            .minCorner = Sz3{ix * subgridSz - (kW / 2), iy * subgridSz - (kW / 2), iz * subgridSz - (kW / 2)},
            .maxCorner = Sz3{std::min((ix + 1) * subgridSz, cartDims[0]) + (kW / 2),
                             std::min((iy + 1) * subgridSz, cartDims[1]) + (kW / 2),
                             std::min((iz + 1) * subgridSz, cartDims[2]) + (kW / 2)}});
        }
      }
    }
  } else if constexpr (NDims == 2) {
    for (Index iy = 0; iy < nB[1]; iy++) {
      for (Index ix = 0; ix < nB[0]; ix++) {
        subgrids.push_back(Subgrid<NDims, VCC>{.minCorner = Sz2{ix * subgridSz - (kW / 2), iy * subgridSz - (kW / 2)},
                                               .maxCorner = Sz2{std::min((ix + 1) * subgridSz, cartDims[0]) + (kW / 2),
                                                                std::min((iy + 1) * subgridSz, cartDims[1]) + (kW / 2)}});
      }
    }
  } else {
    for (Index ix = 0; ix < nB[0]; ix++) {
      subgrids.push_back(Subgrid<NDims, VCC>{.minCorner = Sz1{ix * subgridSz - (kW / 2)},
                                             .maxCorner = Sz1{std::min((ix + 1) * subgridSz, cartDims[0]) + (kW / 2)}});
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
      for (Index ii = 0; ii < NDims; ii++) {
        xyz[ii] = p[ii] * osamp + center[(size_t)ii];
      }
      auto const gp = nearby(xyz);
      auto const off = xyz - gp.template cast<float>();

      std::array<int16_t, NDims> ijk;
      for (int ii = 0; ii < NDims; ii++) {
        ijk[(size_t)ii] = static_cast<int16_t>(Wrap(gp[ii], cartDims[(size_t)ii]));
      }
      cart.push_back(ijk);
      offset.push_back(off);
      noncart.push_back(NoncartesianIndex{.trace = is, .sample = ir});

      // Calculate subgrid
      Index ib = 0;
      for (Index ii = NDims - 1; ii >= 0; ii--) {
        ib = ib * nB[ii] + ((Index)ijk[ii] / subgridSz);
      }
      subgrids[(size_t)ib].indices.push_back(index);
      index++;
    }
  }
  Log::Print("Ignored {} invalid trajectory points", invalids);

  std::vector<Subgrid<NDims, VCC>> chunked;
  for (auto &subgrid : subgrids) {
    if (subgrid.count() > splitSize) {
      for (auto const indexChunk : tl::views::chunk(subgrid.indices, splitSize)) {
        chunked.push_back(Subgrid<NDims, VCC>{.minCorner = subgrid.minCorner,
                                              .maxCorner = subgrid.maxCorner,
                                              .indices = indexChunk | tl::to<std::vector<int32_t>>()});
      }
      subgrid.indices.clear();
    }
  }

  auto const eraseCount = std::erase_if(subgrids, [](Subgrid<NDims, VCC> const &b) { return b.empty(); });
  subgrids.insert(subgrids.end(), chunked.begin(), chunked.end());
  Log::Print("Added {} extra, removed {} empty subgrids, {} remaining", chunked.size(), eraseCount, subgrids.size());
  Log::Print("Total points {}",
             std::accumulate(subgrids.begin(), subgrids.end(), 0UL,
                             [](size_t sum, Subgrid<NDims, VCC> const &b) { return b.indices.size() + sum; }));
  sortedIndices = sort(cart);
}

template struct Mapping<1, false>;
template struct Mapping<2, false>;
template struct Mapping<3, false>;

template struct Mapping<1, true>;
template struct Mapping<2, true>;
template struct Mapping<3, true>;

} // namespace rl
