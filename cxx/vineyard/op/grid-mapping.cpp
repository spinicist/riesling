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

template <int ND>
auto CalcMapping(TrajectoryN<ND> const &traj, float const nomOS, Index const kW, Index const sgSz) -> CalcMapping_t<ND>
{
  auto const  nomDims = traj.matrix();
  auto const  cartDims = Mul(nomDims, nomOS);
  Sz2 const   noncartDims{traj.nSamples(), traj.nTraces()};
  float const osamp = cartDims[0] / (float)traj.matrix()[0];
  Log::Print("Mapping samples {} traces {} OS {} Matrix {} Grid {}", traj.nSamples(), traj.nTraces(), nomOS, nomDims, cartDims);

  std::fesetround(FE_TONEAREST);
  auto const               center = Div(cartDims, 2);
  Index                  valid = 0;
  Index                    invalids = 0;
  std::vector<Mapping<ND>> mappings;
  for (int32_t it = 0; it < traj.nTraces(); it++) {
    for (int16_t is = 0; is < traj.nSamples(); is++) {
      Re1 const p = traj.point(is, it);
      if (!B0(p.isfinite().all())()) {
        invalids++;
        continue;
      }
      Eigen::Array<float, ND, 1> xyz;
      for (Index ii = 0; ii < ND; ii++) {
        xyz[ii] = p[ii] * osamp + center[(size_t)ii];
      }
      auto const                       gp = nearby(xyz);
      Eigen::Array<float, ND, 1> const off = xyz - gp.template cast<float>();

      std::array<int16_t, ND> ijk;
      for (int ii = 0; ii < ND; ii++) {
        ijk[(size_t)ii] = static_cast<int16_t>(Wrap(gp[ii], cartDims[(size_t)ii]));
      }

      Sz<ND> subgrid;
      for (Index id = 0; id < ND; id++) {
        auto const m = sgSz * (ijk[id] / sgSz);
        subgrid[id] = m - (kW / 2);
        ijk[id] -= subgrid[id];
      }
      mappings.push_back(Mapping<ND>{.cart = ijk, .sample = is, .trace = it, .offset = off, .subgrid = subgrid});
      valid++;
    }
  }
  Log::Print("Ignored {} invalid trajectory points, {} remaing", invalids, valid);

  std::sort(mappings.begin(), mappings.end(), [](Mapping<ND> const &a, Mapping<ND> const &b) {
    // First compare on subgrids
    for (size_t di = 0; di < ND; di++) {
      size_t id = ND - 1 - di;
      if (a.subgrid[id] < b.subgrid[id]) { return true; }
      else if (b.subgrid[id] < a.subgrid[id]) {return false; }
    }
    // Then compare on ijk location
    for (size_t di = 0; di < ND; di++) {
      size_t id = ND - 1 - di;
      if (a.cart[id] < b.cart[id]) { return true; }
      else if (b.cart[id] < a.cart[id]) {return false; }
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
