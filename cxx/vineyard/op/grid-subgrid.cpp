#include "grid-subgrid.hpp"

namespace rl {

template <int NDims, bool VCC>
auto Subgrid<NDims, VCC>::empty() const -> bool
{
  return indices.empty();
}

template <int NDims, bool VCC>
auto Subgrid<NDims, VCC>::count() const -> Index
{
  return indices.size();
}

template <int NDims, bool VCC>
auto Subgrid<NDims, VCC>::size() const -> Sz<NDims>
{
  Sz<NDims> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

template struct Subgrid<1, false>;
template struct Subgrid<2, false>;
template struct Subgrid<3, false>;

template struct Subgrid<1, true>;
template struct Subgrid<2, true>;
template struct Subgrid<3, true>;

} // namespace rl
