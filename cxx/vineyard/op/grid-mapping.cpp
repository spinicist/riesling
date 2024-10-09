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

template <int ND> inline auto Sz2Vector(Sz<ND> const &sz) -> Eigen::Vector<float, ND>
{
  Eigen::Vector<float, ND> a;
  for (Index ii = 0; ii < ND; ii++) {
    a[ii] = sz[ii];
  }
  return a;
}

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &traj, float const nomOS, Index const kW, Index const sgSz) -> CalcMapping_t<ND>
{
  std::fesetround(FE_TONEAREST);
  std::vector<Mapping<ND>> mappings;
  auto const               nomDims = traj.matrix();
  auto const               cartDims = MulToEven(nomDims, nomOS);
  auto const               kd = Sz2Vector(cartDims);
  Sz2 const                noncartDims{traj.nSamples(), traj.nTraces()};
  float const              osamp = cartDims[0] / (float)traj.matrix()[0];
  Log::Print("Grid", "Mapping samples {} traces {} OS {} Matrix {} Grid {}", traj.nSamples(), traj.nTraces(), nomOS, nomDims,
             cartDims);

  using Vec = Eigen::Vector<float, ND>;
  Vec const k0 = Sz2Vector(cartDims) / 2;
  Index     valid = 0;
  Index     invalids = 0;

  for (int32_t it = 0; it < traj.nTraces(); it++) {
    for (int16_t is = 0; is < traj.nSamples(); is++) {
      auto const p = traj.point(is, it);
      if ((p.array() != p.array()).any()) {
        invalids++;
        continue;
      }
      Vec const k = p * osamp + k0;
      Vec const ki = Nearby(k);
      Vec const ko = k - ki;
      if ((ki.array() < 0.f).any() || (ki.array() >= kd.array()).any()) {
        invalids++;
        continue;
      }

      std::array<int16_t, ND> ijk;
      for (int ii = 0; ii < ND; ii++) {
        ijk[(size_t)ii] = static_cast<int16_t>(ki[ii]);
      }

      Sz<ND> subgrid;
      for (Index id = 0; id < ND; id++) {
        subgrid[id] = ijk[id] / sgSz;
        ijk[id] -= (sgSz * subgrid[id]) - (kW / 2);
      }
      mappings.push_back(Mapping<ND>{.cart = ijk, .sample = is, .trace = it, .offset = ko, .subgrid = subgrid});
      valid++;
    }
  }
  Log::Print("Grid", "Ignored {} invalid trajectory points, {} remaing", invalids, valid);

  std::sort(mappings.begin(), mappings.end(), [](Mapping<ND> const &a, Mapping<ND> const &b) {
    // First compare on subgrids
    for (size_t di = 0; di < ND; di++) {
      size_t id = ND - 1 - di;
      if (a.subgrid[id] < b.subgrid[id]) {
        return true;
      } else if (b.subgrid[id] < a.subgrid[id]) {
        return false;
      }
    }
    // Then compare on ijk location
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

  return {mappings, noncartDims, cartDims};
}

template struct Mapping<1>;
template struct Mapping<2>;
template struct Mapping<3>;

template auto CalcMapping<1>(TrajectoryN<1> const &traj, float const nomOS, Index const kW, Index const sgSz)
  -> CalcMapping_t<1>;
template auto CalcMapping<2>(TrajectoryN<2> const &traj, float const nomOS, Index const kW, Index const sgSz)
  -> CalcMapping_t<2>;
template auto CalcMapping<3>(TrajectoryN<3> const &traj, float const nomOS, Index const kW, Index const sgSz)
  -> CalcMapping_t<3>;

} // namespace rl
