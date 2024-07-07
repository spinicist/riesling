#include "grid-subgrid.hpp"

namespace rl {

template <int NDims>
auto Subgrid<NDims>::empty() const -> bool
{
  return indices.empty();
}

template <int NDims>
auto Subgrid<NDims>::count() const -> Index
{
  return indices.size();
}

template <int NDims>
auto Subgrid<NDims>::size() const -> Sz<NDims>
{
  Sz<NDims> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

template struct Subgrid<1>;
template struct Subgrid<2>;
template struct Subgrid<3>;

} // namespace rl
