#include "bucket.hpp"

namespace rl {

template <int NDims>
auto Bucket<NDims>::empty() const -> bool
{
  return indices.empty();
}

template <int NDims>
auto Bucket<NDims>::size() const -> Index
{
  return indices.size();
}

template <int NDims>
auto Bucket<NDims>::bucketSize() const -> Sz<NDims>
{
  Sz<NDims> sz;
  std::transform(maxCorner.begin(), maxCorner.end(), minCorner.begin(), sz.begin(), std::minus());
  return sz;
}

template <int NDims>
auto Bucket<NDims>::bucketStart() const -> Sz<NDims>
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
auto Bucket<NDims>::gridStart() const -> Sz<NDims>
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
auto Bucket<NDims>::sliceSize() const -> Sz<NDims>
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

template struct Bucket<1>;
template struct Bucket<2>;
template struct Bucket<3>;

} // namespace rl
