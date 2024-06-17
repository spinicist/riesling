#include "pad.hpp"

#include "op/pad.hpp"

namespace rl {

template <typename Scalar, int N>
auto Pad(Eigen::Tensor<Scalar, N> const &t, Sz<N> const oshape) -> Eigen::Tensor<Scalar, N>
{
  TOps::Pad<Scalar, N> op(t.dimensions(), oshape);
  return op.forward(t);
}

template <typename Scalar, int N>
auto Crop(Eigen::Tensor<Scalar, N> const &t, Sz<N> const oshape) -> Eigen::Tensor<Scalar, N>
{
  TOps::Crop<Scalar, N> op(t.dimensions(), oshape);
  return op.forward(t);
}

template auto Pad<Cx, 1>(Cx1 const &, Sz1 const) -> Cx1;
template auto Pad<Cx, 2>(Cx2 const &, Sz2 const) -> Cx2;
template auto Pad<Cx, 3>(Cx3 const &, Sz3 const) -> Cx3;
template auto Pad<Cx, 5>(Cx5 const &, Sz5 const) -> Cx5;
template auto Crop<Cx, 1>(Cx1 const &, Sz1 const) -> Cx1;
template auto Crop<Cx, 2>(Cx2 const &, Sz2 const) -> Cx2;
template auto Crop<Cx, 3>(Cx3 const &, Sz3 const) -> Cx3;
template auto Crop<Cx, 5>(Cx5 const &, Sz5 const) -> Cx5;
} // namespace rl
