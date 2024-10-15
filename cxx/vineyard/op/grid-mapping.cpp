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
template <typename T> inline decltype(auto) Nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return std::nearbyint(e); });
}

template <int ND> inline auto Sz2Array(Sz<ND> const &sz) -> Eigen::Array<float, ND, 1>
{
  Eigen::Array<float, ND, 1> a;
  for (Index ii = 0; ii < ND; ii++) {
    a[ii] = sz[ii];
  }
  return a;
}

template <int ND>
inline auto SubgridIndex(Eigen::Array<int16_t, ND, 1> const &sg, Eigen::Array<int16_t, ND, 1> const &ngrids) -> Index
{
  Index ind = 0;
  Index stride = 1;
  for (Index ii = 0; ii < ND; ii++) {
    ind += stride * sg[ii];
    // fmt::print(stderr, "ii {} ind {} stride {} sg[ii] {}\n", ii, ind, stride, sg[ii]);
    stride *= ngrids[ii];
  }
  return ind;
}

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &traj, Sz<ND> const &shape, Sz<ND> const &oshape, Index const kW, Index const sgSz)
  -> std::vector<SubgridMapping<ND>>
{
  std::fesetround(FE_TONEAREST);
  std::vector<Mapping<ND>> mappings;

  using Arrayf = Mapping<ND>::template Array<float>;
  using Arrayi = Mapping<ND>::template Array<int16_t>;

  Arrayf const mat = Sz2Array(traj.matrix());
  Arrayf const omat = Sz2Array(oshape);
  Arrayf const osamp = omat / mat;
  Log::Print("Grid", "Mapping matrix {} over-sampled matrix {} over-sampling {}", mat.transpose(), omat.transpose(),
             osamp.transpose());

  Arrayf const k0 = omat / 2;
  Index        valid = 0;
  Index        invalids = 0;
  Arrayi const nSubgrids = (omat / sgSz).ceil().template cast<int16_t>();
  Index const  nTotal = nSubgrids.prod();

  std::vector<SubgridMapping<ND>> subs(nTotal);
  for (int32_t it = 0; it < traj.nTraces(); it++) {
    for (int16_t is = 0; is < traj.nSamples(); is++) {
      Arrayf const p = traj.point(is, it);
      if ((p != p).any()) {
        invalids++;
        continue;
      }
      Arrayf const k = p * osamp + k0;
      Arrayf const ki = Nearby(k);
      if ((ki < 0.f).any() || (ki >= omat).any()) {
        invalids++;
        continue;
      }
      Arrayf const ko = k - ki;
      Arrayi const ksub = (ki / sgSz).floor().template cast<int16_t>();
      Arrayi const kint = ki.template cast<int16_t>() - (ksub * sgSz) + (kW / 2);
      Index const  sgind = SubgridIndex(ksub, nSubgrids);
      subs[sgind].corner = ksub;
      subs[sgind].mappings.push_back(Mapping<ND>{.cart = kint, .sample = is, .trace = it, .offset = ko});
      valid++;
    }
  }
  Log::Print("Grid", "Ignored {} invalid trajectory points, {} remaing", invalids, valid);
  auto const eraseCount = std::erase_if(subs, [](auto const &s) { return s.mappings.empty(); });
  Log::Print("Grid", "Removed {} empty subgrids, {} remaining", eraseCount, subs.size());
  Log::Print("Grid", "Sorting subgrids");
  std::sort(subs.begin(), subs.end(),
            [](SubgridMapping<ND> const &a, SubgridMapping<ND> const &b) { return a.mappings.size() > b.mappings.size(); });
  Log::Print("Grid", "Sorting mappings");
  for (auto &s : subs) {
    std::sort(s.mappings.begin(), s.mappings.end(), [](Mapping<ND> const &a, Mapping<ND> const &b) {
      // Compare on ijk location
      for (size_t di = 0; di < ND; di++) {
        size_t id = ND - 1 - di;
        if (a.cart[id] < b.cart[id]) {
          return true;
        } else if (b.cart[id] < a.cart[id]) {
          return false;
        }
      }
      return false;
    });
  }

  return subs;
}

template struct Mapping<1>;
template struct Mapping<2>;
template struct Mapping<3>;

template auto CalcMapping<1>(TrajectoryN<1> const &, Sz<1> const &, Sz<1> const &, Index const, Index const)
  -> std::vector<SubgridMapping<1>>;
template auto CalcMapping<2>(TrajectoryN<2> const &, Sz<2> const &, Sz<2> const &, Index const, Index const)
  -> std::vector<SubgridMapping<2>>;
template auto CalcMapping<3>(TrajectoryN<3> const &, Sz<3> const &, Sz<3> const &, Index const, Index const)
  -> std::vector<SubgridMapping<3>>;

} // namespace rl
